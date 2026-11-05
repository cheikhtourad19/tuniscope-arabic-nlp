"""Lightweight-kernel bridge for explicit native BERT fine-tuning modes."""
import hashlib,json,os,signal,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run_tool(script,*args):
    if script not in {'run_encoder_finetuning.py','verify_encoder_finetuning.py','plot_encoder_finetuning.py','encoder_finetuning_feasibility.py'}:raise ValueError('Unexpected fine-tuning tool')
    env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env['MPLBACKEND']='Agg'
    command=['sh',str(ROOT/'scripts/encoder_uv.sh'),'run','--locked','--offline','python','-u',str(ROOT/'scripts'/script),*map(str,args)]
    child=subprocess.Popen(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1,start_new_session=True)
    try:
        for line in child.stdout:print(line,end='',flush=True)
        code=child.wait()
    except BaseException:
        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        child.wait();raise
    finally:child.stdout.close()
    if code:raise subprocess.CalledProcessError(code,command)


def selected(directory,expected_mode):
    directory=Path(directory);directory=(ROOT/directory).resolve() if not directory.is_absolute() else directory.resolve()
    if (ROOT/'outputs/encoder_finetuning').resolve() not in directory.parents:raise ValueError('Choose an encoder fine-tuning run')
    d=json.loads((directory/'results.json').read_text())
    if d['mode']!=expected_mode or d['test_loaded'] is not False:raise ValueError('Explicit replay mode mismatch; no smoke/full fallback')
    seeds=[42] if expected_mode=='smoke' else [13,42,2026]
    arms=['frozen','finetuned'] if d['regime']=='paired' else [d['regime']]
    expected={(c,s,a) for c in ('tsac','madar') for s in seeds for a in arms}
    if len(d['results'])!=len(expected) or {(r['corpus'],r['seed'],r['regime']) for r in d['results']}!=expected:raise ValueError('Incomplete run')
    return directory,d


def notebook_run(mode,regime='paired',replay_directory=None,replay_mode='full'):
    if mode not in {'full','smoke','replay'} or regime not in {'paired','frozen','finetuned'}:raise ValueError('Invalid mode/regime')
    if mode=='replay':
        directory,_=selected(replay_directory,replay_mode);run_tool('verify_encoder_finetuning.py',directory);return selected(directory,replay_mode)
    parent=ROOT/'outputs/encoder_finetuning/notebooks';identifier=uuid.uuid4().hex;smoke=parent/(identifier+'_smoke')
    run_tool('run_encoder_finetuning.py','--mode','smoke','--regime','paired','--output',smoke)
    run_tool('verify_encoder_finetuning.py',smoke)
    run_tool('encoder_finetuning_feasibility.py','--smoke-run',smoke)
    if mode=='smoke':return selected(smoke,'smoke')
    budget=json.loads((smoke/'budget_estimate.json').read_text())
    print(f"Estimation depuis le petit essai : comparaison appariée ≈{budget['estimated_paired_minutes']:.0f} min ; ce n'est pas une durée garantie.",flush=True)
    target=parent/(identifier+'_full')
    run_tool('run_encoder_finetuning.py','--mode','full','--regime',regime,'--smoke-run',smoke,'--output',target)
    run_tool('verify_encoder_finetuning.py',target);return selected(target,'full')


def figures(directory):
    target=directory/'figures'
    if not target.exists():run_tool('plot_encoder_finetuning.py',directory)
    p=json.loads((target/'provenance.json').read_text());d=json.loads((directory/'results.json').read_text())
    expected={r['checkpoint']+'.png' for r in d['results']}
    if p['source_results_sha256']!=digest(directory/'results.json') or set(p['files'])!=expected:raise ValueError('Wrong figure source/set')
    result=[]
    for n,h in p['files'].items():
        if Path(n).name!=n or digest(target/n)!=h:raise ValueError('Figure checksum mismatch')
        result.append(target/n)
    return result
