import urllib.request
hdrs = {'User-Agent': 'Mozilla/5.0'}
for f in ('scfc/anatomical_connectivity.py', 'figures/figure_1.py'):
    url = f'https://raw.githubusercontent.com/mhturner/SC-FC/master/{f}'
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode('utf-8', 'replace')
        open('scfc_src_' + f.replace('/', '_'), 'w', encoding='utf-8').write(txt)
        print('fetched', f, len(txt))
    except Exception as ex:
        print(f, 'ERR', ex)
