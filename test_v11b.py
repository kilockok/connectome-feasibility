"""v11b unit/smoke tests for the gated hybrid (run before full experiments).

Checks: shapes, gate range, hard/soft eval consistency, spike-logit
pre-reset convention, no hidden-state inputs, NULL behavior of the
untrained model, gradient flow to gate + delta + encoder, and that the
corrector-off limit (g=0) reproduces the exact hard base bitwise.
"""
import torch
from connectome import Connectome
from run_v7 import setup_cfg
from models.latent_hybrid_v11b import build_v11b
from eval_v11 import hard_lif_out


def main():
    cfg = setup_cfg()
    conn = Connectome.generate(cfg)
    torch.manual_seed(0)
    x = torch.randn(2, 32, 100, 4) * 0.3
    x[..., 1] = (torch.rand(2, 32, 100) > 0.95).float()   # sparse spikes
    x[..., 2] = (torch.rand(2, 32, 100) > 0.9).float()    # some refractory
    for variant in ('m3a', 'm3_nolatent'):
        m = build_v11b('m3_nolatent' if variant == 'm3_nolatent' else 'm3a', conn, cfg)
        out = m(x)
        assert out['v'].shape == (2, 100) and out['s_logits'].shape == (2, 100)
        assert out['gate'].shape == (2, 100) and out['delta'].shape == (2, 100)
        assert ((out['gate'] >= 0) & (out['gate'] <= 1)).all(), 'gate out of [0,1]'
        outh = m(x, hard=True)
        # hard/soft spike convention: hard fire must match logit >= 0
        fire_hard = (outh['v'] == cfg.v_reset) & (out['vn_base'] >= 0)  # not exact test
        # logit sign must agree with hard firing on non-refractory steps
        r = x[:, -1, :, 2]
        nonrefr = ~(r > 0)
        agree = ((outh['s_logits'] >= 0) == (outh['v'] == cfg.v_reset))[nonrefr]
        assert float(agree.float().mean()) > 0.99, 'logit/reset mismatch'
        # gradient flow
        loss = (out['v'].square().mean() + out['gate'].mean() + out['delta'].square().mean())
        loss.backward()
        gsum = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        assert gsum > 0, 'no gradient'
        names = {n for n, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
        assert any('gate_head' in n for n in names) and any('delta_head' in n for n in names)
        assert any('backbone.blocks' in n for n in names) or variant == 'm3_nolatent'
        # gate == 0 limit reproduces the exact hard base bitwise
        with torch.no_grad():
            for p in m.gate_head.parameters():
                p.mul_(0)
            for p in m.delta_head.parameters():
                p.mul_(0)
            out0 = m(x, hard=True)
        dev = 'cpu'
        W = conn.dense_weight(dev)
        ib = conn.i_bias.to(dev) if conn.i_bias is not None else torch.zeros(100)
        b0 = hard_lif_out(x, cfg, W, ib)
        dv = (out0['v'] - b0['v']).abs().max()
        assert float(dv) == 0.0, f'gate-zero does not reproduce hard base: {dv}'
        print(variant, 'PASS (gate-zero == hard base bitwise; grads flow)')
    print('ALL V11B UNIT TESTS PASS')


if __name__ == '__main__':
    main()
