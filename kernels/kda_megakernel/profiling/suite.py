"""Orchestrate verification, ordinary timing, CUPTI and NCU in separate processes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from ..cases.inputs import INPUTS_PATH, MANIFEST_PATH
from ..runner import BASELINES, ROOT, identity, load, write
from .counters import NCU_METRICS, query_metrics


def collect_suite(args, experiment):
    """Keep numerical, ordinary timing and replay evidence in separate runs."""
    if not args.tag:
        raise ValueError('A suite requires a unique --tag to preserve historical evidence')
    manifest = args.manifest.resolve()
    computation = [ROOT/'reference.py', ROOT/'baseline.py', INPUTS_PATH,
                   manifest, *experiment.glob('v[0-9]*/implementation.py')]
    frozen = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in computation}
    summary_path = experiment/'reports'/('collection'+args.tag+'.json')
    if summary_path.exists() and not args.resume:
        raise FileExistsError('Suite tag already exists; use a fresh tag or --resume')
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else None
    metadata = identity(manifest, experiment=experiment)
    print(f"GPU: {metadata['device']} (CUDA {metadata['cuda_device_index']}, UUID {metadata['uuid']})", flush=True)
    if previous and (previous['frozen_computation'] != frozen or previous['implementations'] != args.impl
                     or previous['case_ids'] != args.cases or previous['metadata']['uuid'] != metadata['uuid']):
        raise ValueError('Resume requires identical computation, cases, implementations and GPU')
    summary = previous or {'status':'running', 'metadata':metadata, 'frozen_computation':frozen,
               'implementations':args.impl, 'case_ids':args.cases, 'stages':{}, 'verification_ground_truth':BASELINES[0]}
    if summary.get('verification_ground_truth') != BASELINES[0]:
        raise ValueError('This suite tag uses historical verification; choose a new tag for baseline verification')
    # The recipe generator also reads active-manifest defaults at import time.
    defaults_identity = {'path':str(MANIFEST_PATH), 'sha256':hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()}
    if summary.get('input_defaults_manifest', defaults_identity) != defaults_identity:
        raise ValueError('Active input defaults changed since this suite started')
    summary['input_defaults_manifest'] = defaults_identity
    summary['status'] = 'running'
    write(summary_path, summary)

    def check_sources():
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in frozen.items()):
            raise RuntimeError('Computation or manifest changed during the suite')
        if hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() != defaults_identity['sha256']:
            raise RuntimeError('Active input defaults changed during the suite')

    env = dict(os.environ, KDA_EXPERIMENT=experiment.name)
    device_arg = f"cuda:{metadata['cuda_device_index']}"
    common = [sys.executable, '-m', 'kernels.kda_megakernel.run', '--manifest', str(manifest),
              '--cases', *args.cases, '--tag='+args.tag, '--device', device_arg]
    for mode in ['verify','timing','profile']:
        check_sources()
        if summary['stages'].get(mode) == 'complete':
            continue
        names = [name for name in args.impl if name != 'reference'] if mode == 'profile' else args.impl
        if not names:
            continue
        command = common + ['--mode',mode,'--impl',*names]
        if mode == 'profile':
            command += ['--warmup','3']
        if mode == 'timing':
            if args.warmup is not None:
                command += ['--warmup',str(args.warmup)]
            if args.repeat is not None:
                command += ['--repeat',str(args.repeat)]
        subprocess.run(command, check=True, env=env)
        summary['stages'][mode] = 'complete'
        write(summary_path, summary)
    if args.hardware:
        selected, query = query_metrics(metadata, NCU_METRICS)
        summary['ncu_metrics'] = selected
        summary['unavailable_ncu_metrics'] = [key for key in NCU_METRICS if key not in selected]
        summary['ncu_metric_query_sha256'] = hashlib.sha256(query.encode()).hexdigest()
        summary['ncu_version'] = subprocess.run(['ncu','--version'],capture_output=True,text=True,check=True).stdout
        for name in args.impl:
            check_sources()
            if name == 'reference':
                # The independent serial oracle is deliberately outside hardware ranking.
                continue
            _, destination = load(name, experiment=experiment)
            target = destination/'profiling'/('ncu'+args.tag)
            target.mkdir(parents=True, exist_ok=True)
            collection_path = target/'collection.json'
            if collection_path.exists() and json.loads(collection_path.read_text())['metrics'] != selected:
                previous_metrics = json.loads(collection_path.read_text())['metrics']
                old_key = hashlib.sha256(json.dumps(previous_metrics).encode()).hexdigest()[:8]
                target.rename(target.with_name(target.name+'-superseded-'+old_key))
                target.mkdir()
            collection = json.loads(collection_path.read_text()) if collection_path.exists() else {
                'metadata':identity(manifest, experiment=experiment), 'metrics':selected, 'cases':{}}
            archive = target/'captures.zip'
            for cid in args.cases:
                check_sources()
                if collection['cases'].get(cid, {}).get('status') == 'complete':
                    continue
                timing = json.loads((destination/('end_to_end'+args.tag+'.json')).read_text())['cases'][cid]
                if timing.get('status') == 'error':
                    collection['cases'][cid] = {'status':'unsupported' if name == 'baseline_flashkda' else 'error',
                                                 'reason':timing['reason']}
                    write(collection_path, collection)
                    continue
                metric_key = hashlib.sha256(json.dumps(selected).encode()).hexdigest()[:8]
                scratch = experiment/'tmp/kda'/('capture'+args.tag)/metric_key/name/cid
                scratch.mkdir(parents=True, exist_ok=True)
                prefix = scratch/'capture'
                info = scratch/'identity.json'
                command = ['ncu','--devices',str(metadata['cuda_device_index']),'--profile-from-start','off','--metrics',','.join(selected),
                           '-o',str(prefix),sys.executable,'-m','kernels.kda_megakernel.run',
                           '--mode','capture','--impl',name,'--case',cid,'--manifest',str(manifest),
                           '--capture-info',str(info),'--device',device_arg]
                print('NCU',name,cid,flush=True)
                with (scratch/'capture.log').open('w') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                record = {'status':'error','command':command,'archive':os.path.relpath(archive, experiment),
                          'member_prefix':cid+'/', 'returncode':result.returncode}
                if result.returncode == 0 and prefix.with_suffix('.ncu-rep').exists():
                    with (scratch/'capture.csv').open('w') as output:
                        subprocess.run(['ncu','--import',str(prefix.with_suffix('.ncu-rep')),'--csv',
                                        '--page','raw','--print-units','base'],stdout=output,check=True)
                    record['status'] = 'complete'
                else:
                    record['reason'] = (scratch/'capture.log').read_text()[-5000:]
                with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED) as bundle:
                    for p in scratch.iterdir():
                        bundle.write(p,cid+'/'+p.name)
                collection['cases'][cid] = record
                write(collection_path,collection)
            summary['stages']['hardware_'+name] = 'complete'
            write(summary_path, summary)
    check_sources()
    summary['status'] = 'complete'
    write(summary_path,summary)
