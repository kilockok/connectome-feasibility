"""Unified latent metrics with explicit aggregation, causal rollout and val-only thresholds."""
import math
import numpy as np
import torch
from latent_data import windows, history_control, sample_indices


def f1_counts(p, t, dims=None):
    tp = (p*t).sum(dim=dims)
    fp = (p*(1-t)).sum(dim=dims)
    fn = ((1-p)*t).sum(dim=dims)
    return 2*tp/(2*tp+fp+fn).clamp(min=1)


def pearson(p, t):
    p, t = p-p.mean(-1, keepdim=True), t-t.mean(-1, keepdim=True)
    den = p.norm(dim=-1)*t.norm(dim=-1)
    return torch.where(den>1e-12, (p*t).sum(-1)/den.clamp(min=1e-12), torch.nan)


def finite_mean(x):
    x = torch.as_tensor(x)
    good = torch.isfinite(x)
    return float(x[good].mean()) if good.any() else None


def summarize(out, y, threshold):
    p, t = (out['s_logits'].sigmoid()>threshold).float(), y[..., 1]
    return dict(spike_f1=float(f1_counts(p, t)),
                v_rmse=float((out['v']-y[...,0]).square().mean().sqrt()),
                r_rmse=float((out['r']-y[...,2]).square().mean().sqrt()),
                spike_rate_true=float(t.mean()), spike_rate_pred=float(p.mean()),
                true_spikes=int(t.sum()), n_windows=len(y))


@torch.no_grad()
def predict_windows(model, data, count=512, seed=8001, control='ordered', batch=32,
                    features=False):
    model.eval()
    bi, ti = sample_indices(data, count, seed)
    outputs, targets, feats, labels = [], [], [], []
    sg = torch.Generator().manual_seed(seed+19)
    for start in range(0, count, batch):
        b, t = bi[start:start+batch], ti[start:start+batch]
        x, y = windows(data, b, t, model.k)
        x = history_control(x, control, sg)
        z = data['z'][b.to(x.device), t.to(x.device)] if model.oracle else None
        out, feat = model(x, z=z, return_features=True)
        outputs.append({k:v.cpu() for k,v in out.items()})
        targets.append(y.cpu())
        if features:
            feats.append(torch.cat((feat.mean(1), feat.std(1, unbiased=False)), -1).cpu())
            labels.append(data['z'][b.to(x.device), t.to(x.device)].cpu())
    out = {k:torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    return out, torch.cat(targets), (torch.cat(feats),torch.cat(labels)) if features else None


def calibrate_threshold(out, y):
    candidates = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    return max(candidates, key=lambda t:summarize(out,y,t)['spike_f1'])


def state_from_output(out, cfg, threshold):
    sp = (out['s_logits'].sigmoid()>threshold).float()
    v = out['v'].clamp(cfg.v_min, cfg.v_th*3)
    r = out['r'].clamp(0,1)
    v = torch.where((sp>.5)|(r>.15), torch.full_like(v,cfg.v_reset),v)
    r = torch.where(sp>.5, torch.ones_like(r),r)
    return torch.stack((v,sp,r), -1)


@torch.no_grad()
def rollout_prediction(model, data, cfg, threshold, n=32, horizon=200, start=32,
                       control='ordered'):
    model.eval()
    n = min(n,len(data['states']))
    horizon = min(horizon, cfg.T-start)
    b = torch.arange(n)
    x,_ = windows(data,b,torch.full((n,),start),model.k)
    sg = torch.Generator().manual_seed(9723)
    pred = []
    for step in range(horizon):
        t = start+step
        z = data['z'][:n,t] if model.oracle else None
        state = state_from_output(model(history_control(x,control,sg),z=z),cfg,threshold)
        # Silence is disabled in the new controlled experiment; preserve support for explicit masks.
        sil = data['silence'][:n]
        resting = torch.zeros_like(state)
        resting[...,0] = cfg.v_rest
        state = torch.where(sil[...,None], resting, state)
        pred.append(state.cpu())
        if step+1<horizon:
            feat = torch.cat((state,data['stimulus'][:n,t+1,:,None]),-1)
            x = torch.cat((x[:,1:],feat[:,None]),1)
    return torch.stack(pred,1), data['states'][:n,start+1:start+horizon+1].cpu()


def first_sustained(mask, length=3):
    run=0
    for i,value in enumerate(mask):
        run = run+1 if value else 0
        if run>=length:
            return i-length+2  # 1-based onset
    return None


def rollout_summary(pred, true, horizons=(5,10,20,50,100,200)):
    rows=[]
    ps,ts=pred[...,1],true[...,1]
    pr,tr=ps.mean(-1),ts.mean(-1)
    # Both zero -> ratio=1; positive prediction / zero truth -> large finite value.
    ratio=(pr+1e-9)/(tr+1e-9)
    low=(ratio<.25)
    high=(ratio>4)
    failures=[]
    for b in range(len(pred)):
        lo=first_sustained(low[b].tolist()); hi=first_sustained(high[b].tolist())
        failures.append(dict(trajectory=b,low=lo,high=hi,
                             dynamic=min([v for v in (lo,hi) if v is not None],default=None)))
    for h in horizons:
        if h>pred.shape[1]:
            continue
        p,t=ps[:,:h],ts[:,:h]
        f=f1_counts(p,t,(1,2))
        active=t.sum((1,2))>0
        rows.append(dict(horizon=h,spike_f1=float(f1_counts(p,t)),
                         spike_f1_macro=float(f.mean()),
                         spike_f1_macro_active=finite_mean(f[active]),
                         active_trajectories=int(active.sum()),
                         v_rmse=float((pred[:,:h,:,0]-true[:,:h,:,0]).square().mean().sqrt()),
                         r_rmse=float((pred[:,:h,:,2]-true[:,:h,:,2]).square().mean().sqrt()),
                         rate_ratio=float((p.mean()+1e-9)/(t.mean()+1e-9)),
                         population_rate_correlation=finite_mean(pearson(pr[:,:h],tr[:,:h])),
                         population_activity_rmse=float((pr[:,:h]-tr[:,:h]).square().mean().sqrt()),
                         population_rate_cosine=finite_mean(torch.nn.functional.cosine_similarity(p.mean(1),t.mean(1),dim=1)),
                         dynamic_failure_fraction=sum(v['dynamic'] is not None and v['dynamic']<=h for v in failures)/len(failures)))
    step_f1=[float(f1_counts(ps[:,i],ts[:,i])) for i in range(ps.shape[1])]
    effective=dict(H_F1_90=next((i+1 for i,f in enumerate(step_f1) if f<.9),None),
                   H_F1_70=next((i+1 for i,f in enumerate(step_f1) if f<.7),None),
                   H_rate=first_sustained(((ratio.mean(0)<.8)|(ratio.mean(0)>1.25)).tolist(),1),
                   H_dynamic_failure=finite_mean(torch.tensor([v['dynamic'] for v in failures if v['dynamic'] is not None],dtype=torch.float32)),
                   dynamic_failure_fraction=sum(v['dynamic'] is not None for v in failures)/len(failures),
                   censoring='null if no crossing; dynamic mean is over failing trajectories only')
    series=dict(rate_pred=pr.mean(0).tolist(),rate_true=tr.mean(0).tolist(),
                f1=step_f1,v_rmse=(pred[...,0]-true[...,0]).square().mean((0,2)).sqrt().tolist(),
                rate_ratio=((pr.mean(0)+1e-9)/(tr.mean(0)+1e-9)).tolist())
    return dict(horizons=rows,effective=effective,failures=failures,series=series)


def paired_bootstrap(a,b,seed=776,n=2000):
    """Trajectory-paired CI; callers must aggregate windows within trajectory first."""
    rng=np.random.default_rng(seed)
    delta=np.asarray(a)-np.asarray(b)
    means=delta[rng.integers(0,len(delta),(n,len(delta)))].mean(1)
    return dict(mean=float(delta.mean()),lo=float(np.quantile(means,.025)),hi=float(np.quantile(means,.975)))
