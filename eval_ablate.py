"""Evaluate graph-ablation checkpoints (sanity checks 1-3).

    python eval_ablate.py [--scale full]

Loads the clean connectome checkpoint plus every available ablated one
(ckpt_connectome_<scale>_ablate-<mode>_seed<S>.pt), evaluates one-step
metrics on identical val / test_seen / test_ood splits, and writes
results/ablation_<scale>.json (+ printed table).

Interpretation guide (full scale):
  clean F1 ~0.95 on test_ood
  shuffle-edges -> ~vanilla level  : the model really reads the topology
  shuffle-weights -> partial drop  : separates topology vs weight贡献
  identity -> still high           : task too easy / leakage — investigate
"""
from __future__ import annotations

import argparse
import json

import torch

from ablate import ABLATIONS
from config import get_config, RESULTS_DIR, CHECKPOINT_DIR, add_common_args
from connectome import get_connectome
from dataset import load_or_generate
from device import get_device
from evaluate import onestep_eval, tune_threshold
from lif import LIFSimulator
from models import build_model


def load_variant(name, tag, cfg, conn, device, scale):
    path = (CHECKPOINT_DIR /
            f"ckpt_{name}_{scale}{tag}_seed{cfg.seed}.pt")
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(name, cfg, conn, device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["epoch"], blob["val_loss"]


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    cfg = get_config(args.scale)
    torch.manual_seed(cfg.seed)
    device = get_device(override=args.device)
    conn = get_connectome(cfg, device)
    sim = LIFSimulator(conn, cfg, device)

    variants = [("clean", "")]
    for mode in ABLATIONS:
        variants.append((mode, f"_ablate-{mode}"))

    models = {}
    for label, tag in variants:
        r = load_variant("connectome", tag, cfg, conn, device, args.scale)
        if r is None:
            print(f"[skip] {label}: no checkpoint")
            continue
        model, ep, vl = r
        models[label] = model
        print(f"[load] {label}: epoch {ep}, val_loss {vl:.4f}")
    if len(models) < 2:
        raise SystemExit("need the clean checkpoint plus at least one "
                         "ablated one; run train.py --graph-ablate first")

    # vanilla transformer as the no-structure reference line
    r = load_variant("transformer", "", cfg, conn, device, args.scale)
    if r is not None:
        models["vanilla_ref"] = r[0]

    data = {split: load_or_generate(split, sim, cfg)
            for split in ("val", "test_seen", "test_ood")}

    out = {}
    for label, model in models.items():
        th = tune_threshold(model, data["val"], cfg, device, n_windows=512)
        out[label] = {"threshold": th}
        for split in ("val", "test_seen", "test_ood"):
            res, _ = onestep_eval(model, data[split], cfg, device, th,
                                  n_windows=1024)
            out[label][split] = res
            print(f"[ablate-onestep] {label:15s} {split:9s} "
                  f"v_mse={res['v_mse']:.4f} f1={res['spike_f1']:.3f}")

    path = RESULTS_DIR / f"ablation_{args.scale}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] {path}")

    print("\n==== summary (spike F1) ====")
    print(f"{'variant':16s} {'val':>7s} {'seen':>7s} {'OOD':>7s}")
    for label, d in out.items():
        print(f"{label:16s} {d['val']['spike_f1']:7.3f} "
              f"{d['test_seen']['spike_f1']:7.3f} "
              f"{d['test_ood']['spike_f1']:7.3f}")


if __name__ == "__main__":
    main()
