import urllib.request, json
hdrs = {'User-Agent': 'Mozilla/5.0'}
for path in ('README.md',):
    for branch in ('main', 'master'):
        try:
            url = f'https://raw.githubusercontent.com/mhturner/SC-FC/{branch}/{path}'
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = r.read().decode('utf-8', 'replace')
            print(f'--- {url} ---')
            print(txt[:3000])
            raise SystemExit
        except SystemExit:
            raise
        except Exception as ex:
            print(branch, path, 'ERR', ex)
