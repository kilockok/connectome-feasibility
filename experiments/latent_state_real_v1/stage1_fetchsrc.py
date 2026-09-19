import urllib.request
hdrs = {'User-Agent': 'Mozilla/5.0'}
for f in ('get_atlas_responses.py', 'scfc/bridge.py', 'scfc/functional_connectivity.py'):
    url = f'https://raw.githubusercontent.com/mhturner/SC-FC/master/{f}'
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=30) as r:
        txt = r.read().decode('utf-8', 'replace')
    name = f.replace('/', '_')
    open(f'scfc_src_{name}', 'w', encoding='utf-8').write(txt)
    print('===', f, len(txt), 'chars ===')
