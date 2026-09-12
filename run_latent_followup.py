"""Matched short continued-training versus tangent, after the core study."""
import json
import numpy as np
import torch
from run_latent_state import ROOT,CKPTS,setup,datasets
from train_latent import train_one
from analyze_latent import evaluate_entry


def main():
    torch.set_num_threads(2)
    cfg,lc,conn=setup('hidden')
    data=datasets('hidden',cfg,lc,conn)
    candidates={k:[json.loads((ROOT/'hidden/training'/f'{k}_seed{s}.json').read_text())
                   for s in (1234,1235,1236)] for k in ('hybrid_k8','hybrid_k16','hybrid_k32')}
    selected=min(candidates,key=lambda k:np.mean([r['best']['one_step'] for r in candidates[k]]))
    plan=dict(selected=selected,selection='minimum mean validation loss across 3 seeds',
              epochs=4,updates_per_epoch=48,lr=1e-4,tangent_sigma=.01,tangent_lambda=.1,
              control='matched continued teacher-forced training from the same checkpoint',
              no_rollout_bptt=True)
    (ROOT/'tangent_plan.json').write_text(json.dumps(plan,indent=2))
    for base in candidates[selected]:
        for tangent in (False,True):
            label=selected+('_tangent' if tangent else '_continued')
            summary=train_one(label,base['seed'],conn,cfg,lc,data,ROOT/'hidden',CKPTS/'hidden',
                              epochs=4,steps=48,init=base['checkpoint'],tangent=tangent)
            evaluate_entry(summary,'hidden',cfg,lc,conn,data)


if __name__=='__main__':
    main()
