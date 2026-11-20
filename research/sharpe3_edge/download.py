"""Download only pinned public CRT inputs; verify before reuse."""
from pathlib import Path
import argparse,hashlib,json,urllib.request


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(__file__).with_name('input_manifest.json');items=json.loads(manifest_path.read_text())
    for item in items:
        path=(dest/item['file']).resolve()
        if path.parent!=dest:raise ValueError('Unsafe input path')
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:continue
        url=f"https://raw.githubusercontent.com/{item['repository']}/{item['ref']}/{item['path']}"
        req=urllib.request.Request(url,headers={'User-Agent':'CRT-IPD-research/1.0'})
        with urllib.request.urlopen(req,timeout=120) as response:data=response.read(item['bytes']+1)
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Input verification failed')
        tmp=path.with_suffix('.part');tmp.write_bytes(data);tmp.replace(path)
    (dest/'manifest.json').write_text(manifest_path.read_text())
    print('Verified',len(items),'pinned files')

if __name__=='__main__':main()
