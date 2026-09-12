"""Teacher-equation diagnostic of information in observed history; not a learned result."""
import json
import torch
from run_latent_state import ROOT,setup,datasets
from latent_probe import probe_metrics,fit_probe,apply_probe
from latent_data import sample_indices,windows
from calibrate_latent import write_csv


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cfg,lc,conn=setup('hidden')
    data=datasets('hidden',cfg,lc,conn)
    w=conn.dense_weight('cuda');bias=conn.i_bias.to('cuda')
    rows=[];instant={}
    for sp,d in data.items():
        # A raw-current linear probe controls for easily decoded instantaneous z information.
        b,t=sample_indices(d,2048 if sp=='train' else 1024,9191)
        x,_=windows(d,b,t,1)
        instant[sp]=(x[:,0].flatten(1).cpu(),d['z'][b.to('cuda'),t.to('cuda')].cpu())
        x0,x1=d['states'][:,:-1],d['states'][:,1:]
        syn=x0[...,1]@w
        free=(x0[...,2]==0)&(x1[...,1]==0)&(x1[...,0]>cfg.v_min+1e-5)&(syn.abs()>1e-5)
        residual=(x1[...,0]-x0[...,0])/cfg.alpha+x0[...,0]-cfg.v_rest-d['stimulus']-bias
        num=(residual*syn*free).sum(-1);den=(syn.square()*free).sum(-1)
        good=den>1e-8
        gain=num/den.clamp(min=1e-8)
        recovered=torch.atanh(((gain-1)/lc.alpha).clamp(-.99999,.99999))
        # At decision t, only transitions through t-1 have been observed.
        estimate=torch.zeros_like(d['z']);available=torch.zeros_like(good)
        for t in range(1,cfg.T):
            estimate[:,t]=lc.rho*torch.where(good[:,t-1],recovered[:,t-1],estimate[:,t-1])
            available[:,t]=good[:,t-1]|available[:,t-1]
        take=slice(32,None)
        metrics=probe_metrics(estimate[:,take].flatten(),d['z'][:,take].flatten())
        rows.append(dict(split=sp,kind='mechanistic_history_filter',**metrics,
                         informative_transition_fraction=float(good[:,take].float().mean()),
                         past_information_available_fraction=float(available[:,take].float().mean()),
                         note='Known teacher equation diagnostic with all past observations; not the trained Transformer and not limited to K32. No current/future z supplied.'))
    probe=fit_probe(instant['train'],instant['val'])
    for sp,(f,z) in instant.items():
        if sp=='train':continue
        rows.append(dict(split=sp,kind='instantaneous_raw_linear_probe',
                         **probe_metrics(apply_probe(probe,f),z),
                         note='Post-hoc linear decoding from flattened current V,S,R,U; diagnostic only.'))
    write_csv(ROOT/'teacher_calibration/history_information.csv',rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
