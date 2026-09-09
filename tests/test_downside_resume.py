import pytest
from research.downside.model import Config

@pytest.mark.parametrize('tamper',['none','config','universe','model'])
def test_resume_cannot_mix_experiments(tamper):
    from research.downside.run import validate_resume
    from research.downside import run
    from dataclasses import asdict
    from pathlib import Path
    import hashlib
    cfg=Config();meta={'config':asdict(cfg),'years':[2024],'null':False,
        'inputs':{'universe':'ndx'},'source_hashes':{n:hashlib.sha256(Path(run.__file__).with_name(n).read_bytes()).hexdigest()
            for n in ('model.py','features.py','policy.py')}}
    if tamper=='config':meta['config']['seed']+=1
    if tamper=='universe':meta['inputs']['universe']='sp500'
    if tamper=='model':meta['source_hashes']['model.py']='wrong'
    if tamper=='none':validate_resume(meta,cfg,'ndx',[2024],False)
    else:
        with pytest.raises(ValueError):validate_resume(meta,cfg,'ndx',[2024],False)
