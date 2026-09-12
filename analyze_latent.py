"""Per-entry evaluation persisted immediately; aggregate figures/tables after completion."""
import argparse
import json
from pathlib import Path
import torch
from run_latent_state import ROOT,setup,datasets
from train_latent import load_model,atomic_save
from eval_latent import predict_windows,summarize,rollout_prediction,rollout_summary
from latent_probe import run_probe,intervention,apply_probe


def evaluate_entry(summary,regime,cfg,lc,conn,data):
    label,seed=summary['label'],summary['seed']
    outpath=ROOT/'eval'/'entries'/f'{regime}_{label}_{seed}.json'
    if outpath.exists():
        return json.loads(outpath.read_text())
    model,blob=load_model(summary,conn,torch.device('cuda'))
    control=summary['control'];threshold=blob['threshold']
    result=dict(regime=regime,label=label,seed=seed,params=summary['params'],
                training_seconds=summary['training_seconds'],checkpoint=summary['checkpoint'],
                epoch=blob['epoch'],threshold=threshold,one_step={},rollout={},probe={},intervention=[])
    raw={}
    for sp in ('val','test_seen','test_ood'):
        o,y,_=predict_windows(model,data[sp],1024,seed=8001,control=control)
        result['one_step'][sp]=summarize(o,y,threshold)
        rp,rt=rollout_prediction(model,data[sp],cfg,threshold,n=32,horizon=200,control=control)
        result['rollout'][sp]=rollout_summary(rp,rt)
        raw[sp]=dict(out=o,target=y,rollout_pred=rp,rollout_true=rt)
    # Inference-only shuffle is distinct from the separately trained shuffled control.
    if label.startswith('hybrid_k') and model.k>1:
        o,y,_=predict_windows(model,data['test_seen'],1024,seed=8001,control='shuffle')
        result['inference_shuffle']=summarize(o,y,threshold)
    if regime=='hidden' and label!='oracle':
        probe,metrics,features=run_probe(model,data,control)
        result['probe']=metrics
        result['intervention']=intervention(model,probe,conn,cfg,lc,threshold,control)
        raw['probe']=probe
        raw['probe_examples']={sp:dict(pred=apply_probe(probe,f),true=z) for sp,(f,z) in features.items() if sp.startswith('test')}
    outpath.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(raw,outpath.with_suffix('.pt'))
    tmp=outpath.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False))
    tmp.replace(outpath)
    print('EVALUATED',regime,label,seed,result['one_step']['test_seen'],result['probe'].get('test_seen'),flush=True)
    del model,raw
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--regime',choices=['markov','hidden'],required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    cfg,lc,conn=setup(args.regime)
    data=datasets(args.regime,cfg,lc,conn)
    for path in sorted((ROOT/args.regime/'training').glob('*.json')):
        evaluate_entry(json.loads(path.read_text()),args.regime,cfg,lc,conn,data)


if __name__=='__main__':
    main()
