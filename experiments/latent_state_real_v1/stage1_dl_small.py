import urllib.request, pathlib, sys, time
hdrs = {'User-Agent': 'Mozilla/5.0'}
files = [
    ('body_ids.csv', 'https://ndownloader.figshare.com/files/26568674'),
    ('CorrelationMatrix_branson.csv', 'https://ndownloader.figshare.com/files/26568677'),
    ('StructuralMatrix_branson.csv', 'https://ndownloader.figshare.com/files/26568680'),
]
for name, url in files:
    dest = pathlib.Path('data') / name
    if dest.exists():
        print('exists', name); continue
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=60) as r, dest.open('wb') as f:
        f.write(r.read())
    print('downloaded', name, dest.stat().st_size / 1e6, 'MB')
