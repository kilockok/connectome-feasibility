import subprocess, sys
PY = r'F:\Code\connectome\feasibility\.venv-cuda\Scripts\python.exe'
jobs = [
    ['eval_v7_natural.py'],
    ['intervention_v7.py'],
    ['diagnose_v7.py'],
]
for j in jobs:
    log = 'logs_v7_' + j[0].replace('.py', '') + '.txt'
    print('=== START', ' '.join(j), flush=True)
    with open(log, 'w') as f:
        r = subprocess.run([PY] + j, cwd=r'F:\Code\connectome\feasibility', stdout=f, stderr=subprocess.STDOUT)
    print('=== EXIT', r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
print('V7 ANALYSIS DONE', flush=True)
