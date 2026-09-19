import json, urllib.request
for name, rec in (('A_kenyon_olfactory', '8166598'), ('B_kenyon_ca_2026', '21821328'), ('C_mbon05_voltage', '18675613')):
    try:
        req = urllib.request.Request(f'https://zenodo.org/api/records/{rec}', headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            meta = json.loads(r.read())
        print('=' * 70)
        print(name, '|', meta.get('metadata', {}).get('title'))
        print('  DOI:', meta.get('doi'), '| license:', (meta.get('metadata', {}).get('license') or {}).get('id'))
        print('  pub:', meta.get('metadata', {}).get('publication_date'))
        desc = (meta.get('metadata', {}).get('description') or '')
        import re
        desc = re.sub('<[^>]+>', ' ', desc)
        print('  desc:', ' '.join(desc.split())[:600])
        for f in meta.get('files', []):
            print('   file: %-50s %8.1f MB' % (f['key'], f['size']/1e6))
    except Exception as ex:
        print(name, 'ERROR:', ex)
