import json, urllib.request
req = urllib.request.Request('https://api.figshare.com/v2/articles/13349282', headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=30) as r:
    meta = json.loads(r.read())
print('title:', meta.get('title'))
print('published:', meta.get('published_date'), 'modified:', meta.get('modified_date'))
print('doi:', meta.get('doi'))
print('license:', meta.get('license'))
print('description (first 500):', (meta.get('description') or '')[:500])
print('files:')
for f in meta.get('files', []):
    print('  %-60s %10.1f MB  %s' % (f['name'], f['size']/1e6, f['download_url']))
import pathlib
pathlib.Path('figshare_article_meta.json').write_text(json.dumps(meta, indent=1))
