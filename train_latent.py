"""Teacher-forced training, deterministic comparison samples, independent checkpoints."""
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from latent_data import sample_indices, windows, history_control
from models.latent_temporal import LatentPredictor, specification
from metrics import compute_loss
from eval_latent import predict_windows, calibrate_threshold, summarize, rollout_prediction, rollout_summary
from calibrate_latent import write_csv


def atomic_save(blob,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(blob,tmp)
    tmp.replace(path)


def loss_number(out,y,cfg):
    return float(compute_loss(out,y,cfg,torch.tensor(cfg.spike_pos_weight))[0])


def train_one(label,seed,conn,cfg,lc,data,outdir,ckptdir,epochs=24,steps=48,batch=16,
              init=None,tangent=False):
    directory=Path(ckptdir)/f'{label}_seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    summary_path=Path(outdir)/'training'/f'{label}_seed{seed}.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    torch.manual_seed(seed)
    base_label=label.removesuffix('_tangent').removesuffix('_continued')
    spec=specification(base_label,conn)
    torch.manual_seed(seed)  # parameter-match search must not affect initialization RNG
    model=LatentPredictor(conn,**spec).to(data['train']['states'].device)
    control={'shuffle':'shuffle','last':'last'}.get(base_label,'ordered')
    if init:
        model.load_state_dict(torch.load(init,map_location=next(model.parameters()).device,weights_only=False)['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4 if init else 3e-4,weight_decay=1e-4)
    pw=torch.tensor(cfg.spike_pos_weight,device=next(model.parameters()).device)
    best=dict(one_step=float('inf'),rollout=-float('inf'),combined=-float('inf'))
    history=[]
    start_epoch=1
    bad=0
    last=directory/'last.pt'
    if last.exists():
        resume=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(resume['state_dict']); opt.load_state_dict(resume['optimizer'])
        history=resume['history'];best=resume['best'];bad=resume['bad'];start_epoch=resume['epoch']+1
        torch.set_rng_state(resume['rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
    start_time=time.monotonic()
    for epoch in range(start_epoch,epochs+1):
        model.train()
        bi,ti=sample_indices(data['train'],steps*batch,100_000+seed*100+epoch)
        sg=torch.Generator().manual_seed(seed*1000+epoch)
        losses=[];grads=[]
        for offset in range(0,len(bi),batch):
            b,t=bi[offset:offset+batch],ti[offset:offset+batch]
            x,y=windows(data['train'],b,t,model.k)
            x=history_control(x,control,sg)
            z=data['train']['z'][b.to(x.device),t.to(x.device)] if model.oracle else None
            opt.zero_grad(set_to_none=True)
            pred=model(x,z=z)
            loss,_=compute_loss(pred,y,cfg,pw)
            if tangent:
                from lif_latent import HiddenStateLIFSimulator
                sim=HiddenStateLIFSimulator(conn,cfg,x.device,lc)
                xp=x.clone()
                xp[:,-1,:,0]+=torch.randn(xp[:,-1,:,0].shape,generator=sg).to(x.device)*.01
                # Teacher retains its genuine hidden state; never passed to the student.
                tz=data['train']['z'][b.to(x.device),t.to(x.device)]
                sil=data['train']['silence'][b.to(x.device)]
                yp=sim.step(xp[:,-1,:,:3],xp[:,-1,:,3],tz,sil)
                yc=sim.step(x[:,-1,:,:3],x[:,-1,:,3],tz,sil)
                # Disable dropout in both finite-difference branches to avoid stochastic mismatch.
                model.eval()
                dp=model(xp)['v']-model(x)['v']
                model.train()
                loss=loss+.1*(dp-(yp[...,0]-yc[...,0])).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {label}/{seed}/{epoch}')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):
                raise RuntimeError('Non-finite gradient')
            opt.step()
            losses.append(float(loss.detach()));grads.append(float(grad))
        vo,vy,_=predict_windows(model,data['val'],512,seed=8001,control=control)
        th=calibrate_threshold(vo,vy)
        vm=summarize(vo,vy,th)
        vl=loss_number(vo,vy,cfg)
        rp,rt=rollout_prediction(model,data['val'],cfg,th,n=8,horizon=50,control=control)
        rr=rollout_summary(rp,rt,(10,20,50))
        r50=rr['horizons'][-1]
        # Fixed before tests; reward F1 and finite rate fidelity, penalize V error and either failure mode.
        rate_score=min(r50['rate_ratio'],1/max(r50['rate_ratio'],1e-9))
        rs=r50['spike_f1']+.2*rate_score-.2*r50['v_rmse']-.1*r50['dynamic_failure_fraction']
        combined=.5*vm['spike_f1']+.5*rs
        row=dict(epoch=epoch,train_loss=sum(losses)/len(losses),val_loss=vl,
                 val_f1=vm['spike_f1'],val_v_rmse=vm['v_rmse'],threshold=th,
                 rollout_score=rs,combined_score=combined,gradient_max=max(grads),
                 elapsed_seconds=time.monotonic()-start_time)
        history.append(row)
        blob=dict(state_dict=model.state_dict(),spec=spec,label=label,seed=seed,epoch=epoch,
                  config=asdict(cfg),latent=asdict(lc),threshold=th,control=control,
                  validation=row,protocol='latent_state_v1_pretransition_v1')
        improved=vl<best['one_step']-1e-6
        for kind,value in (('one_step',vl),('rollout',rs),('combined',combined)):
            if (value<best[kind]) if kind=='one_step' else (value>best[kind]):
                best[kind]=value
                atomic_save(blob,directory/f'best_{kind}.pt')
        bad=0 if improved else bad+1
        atomic_save(dict(**blob,optimizer=opt.state_dict(),history=history,best=best,bad=bad,
                         rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),last)
        write_csv(Path(outdir)/'training'/f'{label}_seed{seed}.csv',history)
        print(f'{label} seed={seed} ep={epoch} loss={vl:.5f} F1={vm["spike_f1"]:.4f} V={vm["v_rmse"]:.4f} rollout={rs:.4f} sec={row["elapsed_seconds"]:.1f}',flush=True)
        if bad>=6:
            break
    summary=dict(label=label,seed=seed,spec=spec,control=control,
                 params=sum(p.numel() for p in model.parameters()),epochs=len(history),
                 training_seconds=sum([history[-1]['elapsed_seconds']]) if history else 0,
                 best=best,checkpoint=str(directory/'best_one_step.pt'),
                 init=str(init) if init else None,tangent=tangent)
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(summary,indent=2))
    del model,opt
    torch.cuda.empty_cache()
    return summary


def load_model(summary,conn,device):
    blob=torch.load(summary['checkpoint'],map_location=device,weights_only=False)
    model=LatentPredictor(conn,**blob['spec']).to(device)
    model.load_state_dict(blob['state_dict'])
    return model.eval(),blob
