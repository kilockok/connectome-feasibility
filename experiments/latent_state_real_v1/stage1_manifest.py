import hashlib, json, pathlib, time
files = {}
for p in sorted(pathlib.Path('data').rglob('*')):
    if p.is_file():
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 22), b''):
                h.update(chunk)
        files[str(p).replace('\\', '/')] = dict(bytes=p.stat().st_size, sha256=h.hexdigest())
manifest = dict(
    created=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    primary_source=dict(
        name='Drosophila central brain connectivity (SC-FC, Turner/Mann/Clandinin)',
        doi='10.6084/m9.figshare.13349282', version='v3 (2023-05-30)',
        url='https://figshare.com/articles/dataset/Drosophila_central_brain_connectivity/13349282',
        license='MIT', analysis_code='https://github.com/mhturner/SC-FC (master)'),
    files=files,
    zenodo_blocked=['8166598', '21821328', '18675613'],
    notes='Zenodo API+web returned HTTP 403 for all three records from this network on 2026-09-19; recorded as access failure, no substitution made.')
pathlib.Path('data_manifest.json').write_text(json.dumps(manifest, indent=1))
print('manifest written,', len(files), 'files hashed')
