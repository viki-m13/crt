"""Fetch only the public immutable CRT inputs; never invent replacement data."""
from __future__ import annotations
import argparse,hashlib,json,re,time,urllib.request
from pathlib import Path


def download(output: Path, manifest: Path | None = None):
    manifest=manifest or Path(__file__).with_name('input_manifest.json')
    text=manifest.read_text();items=json.loads(text)
    output.mkdir(parents=True,exist_ok=True)
    seen=set()
    for item in items:
        name=item['file'];ref=item['ref'];source=item['path']
        if (item['repository']!='viki-m13/crt' or not re.fullmatch(r'[a-f0-9]{40}',ref)
                or Path(name).name!=name or name in seen or name in ('.','..')
                or '..' in Path(source).parts or source.startswith('/')
                or not re.fullmatch(r'[a-f0-9]{64}',item['sha256'])):
            raise ValueError('Untrusted or duplicate manifest entry')
        seen.add(name);dest=output/name
        def verified():
            return (dest.exists() and dest.stat().st_size==item['bytes']
                    and hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256'])
        if verified():
            print('verified cached',name,flush=True);continue
        url=f'https://raw.githubusercontent.com/viki-m13/crt/{ref}/{source}'
        temporary=output/(name+'.partial')
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'CRT-failure-first-research/1'})
                with urllib.request.urlopen(req,timeout=90) as response,temporary.open('wb') as handle:
                    total=0;digest=hashlib.sha256()
                    while chunk:=response.read(1024*1024):
                        total+=len(chunk)
                        if total>item['bytes']:raise ValueError('Source exceeds pinned size')
                        handle.write(chunk);digest.update(chunk)
                if total!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                    raise ValueError('Downloaded bytes do not match pinned input')
                temporary.replace(dest);print('downloaded and verified',name,total,flush=True);break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2**attempt)
    (output/'manifest.json').write_text(text)
    return len(items)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();download(args.output)
