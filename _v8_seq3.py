import subprocess, sys
PY = r'F:\Code\connectome\feasibility\.venv-cuda\Scripts\python.exe'
jobs = [
    ['sweep_decoder_v8.py', '--part', 'sweep'],
    ['sweep_decoder_v8.py', '--part', 'decoder'],
]
for j in jobs:
    log = 'logs_v8_' + j[0].replace('.py', '') + '_' + j[2] + '.txt'
    print('=== START', ' '.join(j), flush=True)
    with open(log, 'w') as f:
        r = subprocess.run([PY] + j, cwd=r'F:\Code\connectome\feasibility', stdout=f, stderr=subprocess.STDOUT)
    print('=== EXIT', r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
print('V8 SWEEP+DECODER DONE', flush=True)
