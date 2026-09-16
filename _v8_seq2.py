import subprocess, sys
PY = r'F:\Code\connectome\feasibility\.venv-cuda\Scripts\python.exe'
for j in (['alias_v8.py'], ['probe_v8.py']):
    log = 'logs_v8_' + j[0].replace('.py', '') + '.txt'
    print('=== START', ' '.join(j), flush=True)
    with open(log, 'w') as f:
        r = subprocess.run([PY] + j, cwd=r'F:\Code\connectome\feasibility', stdout=f, stderr=subprocess.STDOUT)
    print('=== EXIT', r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
print('V8 ALIAS+PROBE DONE', flush=True)
