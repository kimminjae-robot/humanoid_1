"""Independently compare every frozen balance weight to its validated checkpoint."""
import argparse,json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);a=p.parse_args();ck=a.checkpoint.resolve();cfg=json.loads((ck.parent/'resolved.json').read_text());base=Path(cfg['agent']['policy']['base_checkpoint'])
    current=torch.load(ck,map_location='cpu',weights_only=False)['model_state_dict'];before=torch.load(base,map_location='cpu',weights_only=False)['model_state_dict']
    deltas={k:float((v-before['actor.'+k.removeprefix('actor.base.')]).abs().max()) for k,v in current.items() if k.startswith('actor.base.')}
    expected={k.removeprefix('actor.') for k in before if k.startswith('actor.')}
    checks={'all_base_parameters_present':{k.removeprefix('actor.base.') for k in deltas}==expected,'base_weights_exactly_unchanged':bool(deltas) and max(deltas.values())==0.,'all_policy_weights_finite':all(bool(torch.isfinite(v).all()) for v in current.values())}
    result={'scope':'independent_CPU_checkpoint_tensor_comparison','checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':sha(ck),'base_checkpoint':str(base.relative_to(ROOT)),'base_checkpoint_sha256':sha(base),'checks':checks,'passed':all(checks.values()),'parameter_max_absolute_differences':deltas,'verifier_sha256':sha(Path(__file__))}
    (ck.parent/('base_balance_verification_'+ck.stem+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='parameter_max_absolute_differences'},indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
