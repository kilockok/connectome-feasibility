"""v9 equivalence tests: single-mechanism configs must reproduce the legacy
v2/v7/v8 teachers bitwise; NULL must equal base LIF; mixture RNG streams
must not perturb the visible layout."""
import torch
from config import Config
from connectome import Connectome
from lif import LIFSimulator
from lif_latent_v2 import HiddenStateLIFSimulatorV2, LatentV2Config
from lif_adapt_v7 import AdaptationLIFSimulator, AdaptConfig
from lif_stp_v8 import STPLIFSimulator, STPConfig, calibrate_norm
from teachers_v9 import (MechanismLIFSimulator, MechSpec, GainParams, AdaptParams,
                         STPParams, OUParams)

cfg = Config(n_neurons=100, T=64, n_train_traj=8)
conn = Connectome.generate(cfg)
dev = torch.device('cuda')
seeds = [cfg.traj_seed('train', i) for i in range(4)]


def maxdiff(a, b):
    return float((a - b).abs().max())


# NULL == base LIF
base = LIFSimulator(conn, cfg, dev).simulate
d_null = MechanismLIFSimulator(conn, cfg, dev, MechSpec()).generate(seeds, 'train')
import dataset
d_base = dataset.generate_batch(seeds, 'train', LIFSimulator(conn, cfg, dev), cfg)
assert maxdiff(d_null['states'][:, 1:], d_base['states']) == 0.0, 'NULL != base'
print('NULL == base LIF: exact')

# GAIN-only == lif_latent_v2
lc = LatentV2Config()
d_gain = MechanismLIFSimulator(conn, cfg, dev, MechSpec(gain=GainParams())).generate(seeds, 'train')
sim_v2 = HiddenStateLIFSimulatorV2(conn, cfg, dev, lc)
d_v2 = sim_v2.generate(seeds, 'train')
assert maxdiff(d_gain['states'], d_v2['states']) == 0.0, 'GAIN != v2'
print('GAIN == lif_latent_v2: exact')

# ADAPT-only == lif_adapt_v7 (v7 uses rng offset +91M)
ap = AdaptParams(rng_offset=91_000_000)
d_adapt = MechanismLIFSimulator(conn, cfg, dev, MechSpec(adapt=ap)).generate(seeds, 'train')
sim_v7 = AdaptationLIFSimulator(conn, cfg, dev, AdaptConfig(beta=ap.beta, tau_a=ap.tau_a, c=ap.c))
d_v7 = sim_v7.generate(seeds, 'train')
assert maxdiff(d_adapt['states'], d_v7['states']) == 0.0, 'ADAPT != v7'
print('ADAPT == lif_adapt_v7: exact')

# STP-only == lif_stp_v8 (strength=1, tau_scale=1)
norm = calibrate_norm(STPLIFSimulator(conn, cfg, dev, STPConfig(enabled=False)), cfg)
sim_v8 = STPLIFSimulator(conn, cfg, dev, STPConfig(enabled=True), norm=norm)
d_v8 = sim_v8.generate(seeds, 'train')
d_stp = MechanismLIFSimulator(conn, cfg, dev, MechSpec(stp=STPParams())).generate(seeds, 'train')
assert maxdiff(d_stp['states'], d_v8['states']) == 0.0, 'STP != v8'
print('STP == lif_stp_v8: exact')

# STP strength=0 == base LIF
d_s0 = MechanismLIFSimulator(conn, cfg, dev, MechSpec(stp=STPParams(strength=0.0))).generate(seeds, 'train')
assert maxdiff(d_s0['states'][:, 1:], d_base['states']) == 0.0, 'STP s=0 != base'
print('STP strength=0 == base: exact')

# mixture keeps the visible layout; OU sigma=0 == base
d_mix = MechanismLIFSimulator(conn, cfg, dev, MechSpec(
    gain=GainParams(alpha=0.0), ou=OUParams(sigma=0.0),
    adapt=AdaptParams(beta=0.0))).generate(seeds, 'train')
assert maxdiff(d_mix['states'][:, 1:], d_base['states']) == 0.0, 'zero-mixture != base'
print('zero-strength mixture == base: exact')

# OU changes dynamics when sigma>0 and stays finite
d_ou = MechanismLIFSimulator(conn, cfg, dev, MechSpec(ou=OUParams(sigma=0.5, tau=16))).generate(seeds, 'train')
diff = maxdiff(d_ou['states'][:, 1:], d_base['states'])
assert 1e-6 < diff < 10.0, diff
assert torch.isfinite(d_ou['states']).all()
print(f'OU active: max|dV|={diff:.3f}, finite')
print('ALL V9 TEACHER TESTS PASSED')

