import urllib.request, json
hdrs = {'User-Agent': 'Mozilla/5.0'}
for url in ('https://male-cns.janelia.org/', 'https://male-cns.janelia.org/about'):
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=20) as r:
            txt = r.read().decode('utf-8', 'replace')
        import re
        body = re.sub('<[^>]+>', ' ', txt)
        body = ' '.join(body.split())
        print(url, '->', len(txt), 'chars')
        print(body[:800])
        print()
    except Exception as ex:
        print(url, 'ERR', ex)
