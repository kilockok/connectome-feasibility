"""Post-hoc linear probes and controlled state tracking. Main model stays frozen."""
import torch
from eval_latent import predict_windows, summarize, first_sustained
from latent_data import windows, history_control
from lif_latent import HiddenStateLIFSimulator


def probe_metrics(pred,true):
    pred,true=pred.double(),true.double()
    den=(true-true.mean()).square().sum()
    pc,tc=pred-pred.mean(),true-true.mean()
    cd=pc.norm()*tc.norm()
    return dict(r2=float(1-(pred-true).square().sum()/den) if den>1e-12 else None,
                correlation=float((pc*tc).sum()/cd) if cd>1e-12 else None,
                mae=float((pred-true).abs().mean()))


def fit_probe(train,val):
    f,z=train; vf,vz=val
    f,vf,z,vz=f.double(),vf.double(),z.double(),vz.double()
    mean=f.mean(0);std=f.std(0).clamp(min=1e-5);zm=z.mean()
    x=(f-mean)/std;v=(vf-mean)/std
    gram=x.T@x;target=x.T@(z-zm)
    best=None
    for ridge in (.01,.1,1.,10.,100.):
        w=torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0],dtype=torch.float64),target)
        error=float(((v@w+zm)-vz).square().mean())
        if best is None or error<best[0]:
            best=error,ridge,w
    return dict(mean=mean,std=std,z_mean=zm,weight=best[2],ridge=best[1])


def apply_probe(probe,f):
    return ((f.double().cpu()-probe['mean'])/probe['std'])@probe['weight']+probe['z_mean']


@torch.no_grad()
def run_probe(model,data,control='ordered'):
    # no_grad outputs are the ONLY training inputs to the independent closed-form linear fit.
    features={sp:predict_windows(model,data[sp],2048 if sp=='train' else 1024,
                                seed=9191,control=control,features=True)[2]
              for sp in ('train','val','test_seen','test_ood')}
    probe=fit_probe(features['train'],features['val'])
    results={sp:dict(**probe_metrics(apply_probe(probe,f),z),ridge=probe['ridge'])
             for sp,(f,z) in features.items() if sp!='train'}
    assert all(p.grad is None for p in model.parameters())
    return probe,results,features


@torch.no_grad()
def intervention(model,probe,conn,cfg,lc,threshold,control='ordered',n=32):
    sim=HiddenStateLIFSimulator(conn,cfg,next(model.parameters()).device,lc)
    rows=[]
    # Fixed jump independent of external pulse times. Uninformative histories remain in the denominator.
    for low,high in ((-1.,1.),(1.,-1.)):
        d=sim.generate(list(range(80_000_000,80_000_000+n)),'test_seen',(128,low,high))
        sg=torch.Generator().manual_seed(32)
        curve=[]
        for delay in range(-16,65):
            end=128+delay
            x,y=windows(d,torch.arange(n),torch.full((n,),end),model.k)
            x=history_control(x,control,sg)
            z=d['z'][:,end] if model.oracle else None
            out,feat=model(x,z=z,return_features=True)
            f=torch.cat((feat.mean(1),feat.std(1,unbiased=False)),-1).cpu()
            predicted=apply_probe(probe,f)
            true=d['z'][:,end].cpu()
            mm=summarize({k:v.cpu() for k,v in out.items()},y.cpu(),threshold)
            row=dict(direction=f'{low:g}_to_{high:g}',delay=delay,
                     probe_mae=float((predicted-true).abs().mean()),
                     probe_mean=float(predicted.mean()),true_z=float(true.mean()),
                     **mm)
            curve.append(row)
        post=[r for r in curve if r['delay']>=0]
        onset=first_sustained([r['probe_mae']<=.25*abs(high-low) for r in post])
        latency=onset-1 if onset is not None else None
        for row in curve:
            row['reacquisition_latency']=latency
            row['latency_definition']='first of 3 consecutive delays with mean absolute z error <=25% of jump; null=censored >64'
            rows.append(row)
    return rows
