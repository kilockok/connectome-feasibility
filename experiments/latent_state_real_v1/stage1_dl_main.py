import urllib.request, pathlib, time, sys
url = 'https://ndownloader.figshare.com/files/26356327'
dest = pathlib.Path('data/data_TurnerMannClandinin.tar.gz')
hdrs = {'User-Agent': 'Mozilla/5.0'}
if dest.exists() and dest.stat().st_size > 240_000_000:
    print('already complete', dest.stat().st_size)
    sys.exit(0)
req = urllib.request.Request(url, headers=hdrs)
t0 = time.time()
with urllib.request.urlopen(req, timeout=120) as r, dest.open('wb') as f:
    total = int(r.headers.get('Content-Length', 0))
    got = 0
    while True:
        chunk = r.read(1 << 22)
        if not chunk:
            break
        f.write(chunk); got += len(chunk)
        el = time.time() - t0
        print(f'{got/1e6:.0f}/{total/1e6:.0f} MB  {got/1e6/max(el,1):.1f} MB/s', flush=True)
print('DONE', dest.stat().st_size)
