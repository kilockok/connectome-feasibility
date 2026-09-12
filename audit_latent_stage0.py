"""Read-only legacy checkpoint regression audit; writes only new-stage output."""
import json
from pathlib import Path
import torch
from config import get_config
from connectome import get_connectome
from dataset import load_or_generate
from lif import LIFSimulator
from models import build_model
from evaluate import tune_threshold, onestep_eval
from reinjection import rollout_reinject
from rollout import rollout_metrics
from rollout_eval_v3 import horizon_metrics_v3


def main():
    torch.set_num_threads(2)
    cfg = get_config('full')
    dev = torch.device('cuda')
    torch.manual_seed(cfg.seed + 3407)
    conn = get_connectome(cfg, dev)
    sim = LIFSimulator(conn, cfg, dev)
    path = Path('results/checkpoints/ckpt_gnn_full_seed1234.pt')
    blob = torch.load(path, map_location=dev, weights_only=False)
    model = build_model('gnn', cfg, conn, dev).eval()
    model.load_state_dict(blob['state_dict'])
    val = load_or_generate('val', sim, cfg)
    th = tune_threshold(model, val, cfg, dev, n_windows=512)
    one, _ = onestep_eval(model, val, cfg, dev, th, n_windows=512)
    data = load_or_generate('test_seen', sim, cfg)
    states, stim = data['states'][:16], data['stimulus'][:16]
    pred = rollout_reinject(model, states, stim, cfg, 1, th,
                           silence_mask=data['silence'][:16])
    true = states[:, cfg.K:]
    horizon = true.shape[1]
    pooled = rollout_metrics(pred, true, [horizon])[horizon]
    macro = horizon_metrics_v3(pred, true, horizon)
    assert one['spike_f1'] > .98, one
    assert pooled['spike_f1'] > .95, pooled
    result = dict(checkpoint=str(path), threshold=th, onestep_val=one,
                  reinjection_k1_pooled=pooled, reinjection_k1_macro=macro,
                  true_spikes=int(true[..., 1].sum()),
                  note='Legacy timing and F1 aggregation preserved; 16 trajectories, full horizon. Initial 4-trajectory/32-step check had only six true spikes (F1=.909); expanded support, no metric changes.')
    out = Path('results/latent_state_v1/stage0')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'baseline_regression.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
