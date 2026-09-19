import json, urllib.request, time
hdrs = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
        'Accept': 'application/json'}
for name, rec in (('A_kenyon_olfactory', '8166598'), ('B_kenyon_ca_2026', '21821328'), ('C_mbon05_voltage', '18675613')):
    ok = False
    for url in (f'https://zenodo.org/api/records/{rec}', f'https://zenodo.org/records/{rec}'):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            meta = json.loads(raw)
            m = meta.get('metadata', meta)
            print('=' * 70)
            print(name, '|', m.get('title'))
            print('  DOI:', meta.get('doi'))
            files = meta.get('files', [])
            for f in files[:20]:
                k = f.get('key') or f.get('filename') or '?'
                sz = f.get('size', 0) / 1e6
                print('   file: %-50s %8.1f MB' % (k, sz))
            ok = True
            break
        except Exception as ex:
            print(name, url, 'ERR', ex)
            time.sleep(2)
    if not ok:
        print(name, 'ALL ENDPOINTS FAILED')
