import subprocess, sys
PY = r'F:\Code\connectome\feasibility\.venv-cuda\Scripts\python.exe'
jobs = [
    ['eval_v8_natural.py'],
    ['run_v8.py', '--regime', 'negctrl', '--seeds', '1234', '1235', '--labels', 'k1', 'k2', 'set', 'ordered', 'event_simple', 'event_rich'],
]
for j in jobs:
    log = 'logs_v8_seq_' + j[0].replace('.py', '') + ('_neg' if 'negctrl' in j else '') + '.txt'
    print('=== START', ' '.join(j), flush=True)
    with open(log, 'w') as f:
        r = subprocess.run([PY] + j, cwd=r'F:\Code\connectome\feasibility', stdout=f, stderr=subprocess.STDOUT)
    print('=== EXIT', r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
print('V8 SEQ DONE', flush=True)
