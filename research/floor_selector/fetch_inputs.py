"""Fetch immutable PUBLIC research inputs, verify size/hash, write atomically."""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


def download(root: Path):
    root.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('input_manifest.json')
    manifest=json.loads(source.read_text())
    for item in manifest:
        path=root/item['file']
        if path.parent.resolve()!=root.resolve():raise ValueError('Unsafe manifest path')
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:
            continue
        url='https://raw.githubusercontent.com/{repository}/{ref}/{path}'.format(**item)
        temp=None
        try:
            with tempfile.NamedTemporaryFile(dir=root,delete=False) as f:
                temp=Path(f.name);digest=hashlib.sha256();size=0
                with urllib.request.urlopen(url,timeout=60) as response:
                    while block:=response.read(1024*1024):
                        size+=len(block)
                        if size>item['bytes']:raise ValueError('Oversized upstream response')
                        f.write(block);digest.update(block)
            if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                raise ValueError('Source size/digest mismatch: '+item['file'])
            os.replace(temp,path);temp=None
            print('Verified',item['file'],flush=True)
        finally:
            if temp is not None:temp.unlink(missing_ok=True)
    (root/'manifest.json').write_text(source.read_text())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    download(parser.parse_args().out)
