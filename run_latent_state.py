"""Resumable controlled study. All compute is intended for the remote CUDA host."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from config import Config
from connectome import Connectome
from lif_latent import HiddenStateLIFSimulator, LatentConfig
from train_latent import train_one, atomic_save

ROOT=Path('results/latent_state_v1')
CKPTS=Path('results/checkpoints/latent_state_v1')


def setup(regime):
    selected=json.loads((ROOT/'teacher_calibration/selection.json').read_text())
    cfg=Config(**selected['config'])
    cfg=replace(cfg,n_train_traj=512,n_val_traj=64,n_test_seen_traj=64,n_test_traj=64)
    lc=LatentConfig(**selected['latent'])
    if regime=='markov':
        lc=replace(lc,alpha=0.)
    conn=Connectome.generate(cfg)
    return cfg,lc,conn


def datasets(regime,cfg,lc,conn):
    path=ROOT/'data'/f'{regime}.pt'
    key=dict(config=asdict(cfg),latent=asdict(lc),protocol='pretransition_v1')
    if path.exists():
        blob=torch.load(path,map_location='cpu',weights_only=False)
        if blob['key']!=key:
            raise ValueError('Dataset cache config mismatch; use a new experiment namespace')
        return {sp:{k:v.to('cuda') for k,v in d.items()} for sp,d in blob['data'].items()}
    sim=HiddenStateLIFSimulator(conn,cfg,torch.device('cuda'),lc)
    all_data={}
    for sp,count in (('train',cfg.n_train_traj),('val',cfg.n_val_traj),
                     ('test_seen',cfg.n_test_seen_traj),('test_ood',cfg.n_test_traj)):
        chunks=[]
        for i in range(0,count,64):
            d=sim.generate([cfg.traj_seed(sp,j) for j in range(i,min(i+64,count))],sp)
            chunks.append({k:v.cpu() for k,v in d.items()})
        all_data[sp]={k:torch.cat([d[k] for d in chunks]) for k in chunks[0]}
        print('data',regime,sp,count,'rate',float(all_data[sp]['states'][...,1].mean()),flush=True)
    atomic_save(dict(key=key,data=all_data),path)
    return {sp:{k:v.to('cuda') for k,v in d.items()} for sp,d in all_data.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['markov','hidden'],required=True)
    p.add_argument('--labels',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int,default=[1234,1235,1236])
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--steps',type=int,default=48)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg,lc,conn=setup(args.stage)
    data=datasets(args.stage,cfg,lc,conn)
    labels=args.labels or (['gnn_k1','hybrid_k32'] if args.stage=='markov' else
                           ['gnn_k1','hybrid_k1','hybrid_k8','hybrid_k16','hybrid_k32','wide','shuffle','last','oracle'])
    snapshot=dict(config=asdict(cfg),latent=asdict(lc),model_seeds=args.seeds,
                  epochs=args.epochs,steps=args.steps,batch=16,labels=labels,
                  training='teacher forced, tangent off, DAgger off',
                  input='X[t],U[t] before transition; target X[t+1]; same endpoint distribution for all K',
                  primary_checkpoint='best_one_step (minimum validation state loss)',
                  zero_spike_F1='0 when both prediction and truth have zero spikes; additionally report active-only macro and pooled F1',
                  precision='float32; TF32 off',dataset_seed=cfg.seed,
                  scope='N=100 controlled feasibility experiment; not a full N=1000 replication')
    out=ROOT/args.stage
    out.mkdir(parents=True,exist_ok=True)
    (out/'config_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    for seed in args.seeds:
        for label in labels:
            train_one(label,seed,conn,cfg,lc,data,out,CKPTS/args.stage,args.epochs,args.steps)
    print('TRAINING STAGE COMPLETE',args.stage,flush=True)


if __name__=='__main__':
    main()
