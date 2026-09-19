import json, urllib.request
hdrs = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request('https://api.github.com/repos/mhturner/SC-FC/git/trees/master?recursive=1', headers=hdrs)
with urllib.request.urlopen(req, timeout=30) as r:
    tree = json.loads(r.read())
for item in tree['tree']:
    if item['path'].endswith(('.py', '.yaml', '.R')):
        print(item['path'])
