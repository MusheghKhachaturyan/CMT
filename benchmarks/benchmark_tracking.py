#!/usr/bin/env python3
"""Reproducible tracker/application wall time, with a separate stage sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import scipy
from CMT import CMT
from tracking_output import draw_and_write
import util


def summary(values):
    values=np.asarray(values,dtype=float)
    if not values.size: return None
    return dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),
                min=float(values.min()),max=float(values.max()))


def capture(path):
    return cv2.VideoCapture(path) if Path(path).is_file() else util.FileVideoCapture(path)


def sequence_parity(cpu_runs, gpu_runs):
    """Compare discrete decisions and floating poses; retain truncation evidence."""
    differences=[]
    reference=cpu_runs[0]['results']
    for run_number,run in enumerate(gpu_runs,1):
        if len(reference)!=len(run['results']):
            differences.append(dict(run=run_number,error='frame count differs'))
            continue
        for frame,(cpu,gpu) in enumerate(zip(reference,run['results']),1):
            exact=all(cpu[name]==gpu[name] for name in ('has_result','active','class_signature'))
            floats=all(np.allclose(cpu[name],gpu[name],rtol=1e-7,atol=1e-6,equal_nan=True)
                       for name in ('center','scale','rotation','corners_float'))
            c=np.asarray(cpu['corners']);g=np.asarray(gpu['corners'])
            corners=np.array_equal(np.isnan(c),np.isnan(g)) and np.all(np.abs(c[np.isfinite(c)]-g[np.isfinite(g)])<=1)
            if not (exact and floats and corners):
                differences.append(dict(run=run_number,frame=frame,discrete_match=exact,
                                        floating_match=floats,corners_match=bool(corners)))
                if len(differences)>=10:return dict(status='failed',first_differences=differences)
    return dict(status='failed' if differences else 'passed',first_differences=differences)


def run_once(args, backend, limit=None, profile=False, output=None):
    app_start=perf_counter()
    cap=capture(args.input)
    if not cap.isOpened(): raise ValueError(f'Unable to open {args.input}')
    try:
        ok,image=cap.read()
        if not ok: raise ValueError('Unable to read initialization frame')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        x,y,w,h=args.bbox
        if min(x,y)<0 or min(w,h)<=0 or x+w>gray.shape[1] or y+h>gray.shape[0]:
            raise ValueError('Bounding box must fit inside initialization frame')
        start=perf_counter()
        tracker=CMT(backend=backend,device=args.device,gpu_memory_mb=args.gpu_memory_mb)
        startup=perf_counter()-start
        tracker.profile.enabled=profile
        tracker.estimate_rotation=args.with_rotation
        tracker.estimate_scale=not args.no_scale
        start=perf_counter()
        tracker.initialise(gray,(x,y),(x+w-1,y+h-1)); tracker.synchronize()
        initialization=perf_counter()-start
        initialization_profile=tracker.profile.snapshot()
        latencies=[]; stages={}; transfers=Counter(); features=[]; memories=[]; routes=Counter(); results=[]
        complete=False
        if output: Path(output).mkdir(parents=True,exist_ok=True)
        tracking_app_start=perf_counter()
        while limit is None or len(latencies)<limit:
            ok,image=cap.read()
            if not ok:
                complete=True
                break
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            tracker.synchronize(); start=perf_counter()
            tracker.process_frame(gray); tracker.synchronize()
            latencies.append(perf_counter()-start)
            if output:
                with tracker.profile.stage('drawing_output'):
                    draw_and_write(tracker,image,image.copy(),str(output),len(latencies))
            snap=tracker.profile.snapshot()
            for name,value in snap['seconds'].items(): stages.setdefault(name,[]).append(value)
            transfers.update(snap['transfers'])
            features.append(snap['features']);memories.append(tracker._backend.memory())
            routes[tracker.geometry_route]+=1
            classes=hashlib.sha256()
            for values in (tracker.active_keypoints,tracker.tracked_keypoints,tracker.outliers):
                classes.update(np.asarray([len(values)],dtype='<i8').tobytes())
                classes.update(values[:,2].astype('<i4').tobytes())
            corners_float=np.full((4,2),np.nan)
            if tracker.has_result:
                offsets=np.stack((tracker.center_to_tl,tracker.center_to_tr,tracker.center_to_br,tracker.center_to_bl))
                corners_float=tracker.center+tracker.scale_estimate*util.rotate(offsets,tracker.rotation_estimate)
            results.append(dict(has_result=bool(tracker.has_result),box=tracker.bb.tolist(),
                                center=tracker.center.tolist(),scale=float(tracker.scale_estimate),
                                rotation=float(tracker.rotation_estimate),active=len(tracker.active_keypoints),
                                class_signature=classes.hexdigest(),corners_float=corners_float.tolist(),
                                corners=np.asarray((tracker.tl,tracker.tr,tracker.br,tracker.bl)).tolist()))
            if len(latencies)%100==0:
                print(f'{backend}: {len(latencies)} frames',file=sys.stderr,flush=True)
        tracking_app_seconds=perf_counter()-tracking_app_start
        if not latencies: raise ValueError('No tracking frames available')
        return dict(backend=tracker._backend.diagnostics(),startup_seconds=startup,
                    initialization_seconds=initialization,initialization_profile=initialization_profile,
                    frames=len(latencies),complete_sequence=complete,profile_enabled=profile,
                    frame_seconds=summary(latencies),tracker_seconds=sum(latencies),
                    tracker_fps=len(latencies)/sum(latencies),application_seconds=tracking_app_seconds,
                    application_fps=len(latencies)/tracking_app_seconds,
                    application_total_seconds=perf_counter()-app_start,
                    application_mode='decode+grayscale+tracking'+('+annotation+PNG+CSV' if output else ''),
                    stages={name:dict(total_seconds=sum(values),seconds=summary(values)) for name,values in stages.items()},
                    transfer_bytes=dict(transfers),feature_counts=features,geometry_routes=dict(routes),
                    max_frame_boundary_pool_used_bytes=max(m['used_bytes'] for m in memories),
                    peak_pool_reserved_bytes=max(m['reserved_bytes'] for m in memories),
                    memory_sampling='frame boundaries; reserved pool retains temporary allocations; live temporary peaks are not measured',
                    memory_by_frame=memories,results=results)
    finally:
        cap.release()


def serializable(value):
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,list): return [serializable(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--bbox',required=True,type=lambda s:tuple(map(int,s.split(','))))
    parser.add_argument('--with-rotation',action='store_true')
    parser.add_argument('--no-scale',action='store_true')
    parser.add_argument('--backends',default='cpu,cupy')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--gpu-memory-mb',type=float,default=256)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--warmup-frames',type=int,default=3)
    parser.add_argument('--profile-frames',type=int,default=30,help='Separate instrumented pass; excluded from throughput')
    parser.add_argument('--max-frames',type=int,help='Smoke test only; marks sequence validation incomplete')
    parser.add_argument('--opencv-threads',type=int,default=cv2.getNumThreads())
    parser.add_argument('--application-output',type=Path,help='Write legacy PNG/CSV output in each measured run')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(args.bbox)!=4 or args.repeats<1 or args.warmup_frames<1 or args.profile_frames<0 or args.opencv_threads<1 or (args.max_frames is not None and args.max_frames<1):
        parser.error('Invalid bbox, frame count, repeat count, or thread count')
    names=args.backends.split(',')
    if any(n not in ('cpu','cupy') for n in names) or len(set(names))!=len(names): parser.error('--backends must be cpu, cupy, or cpu,cupy')
    cv2.setNumThreads(args.opencv_threads)
    fingerprint=None
    if Path(args.input).is_file():
        digest=hashlib.sha256()
        with open(args.input,'rb') as source:
            for chunk in iter(lambda:source.read(1024**2),b''):digest.update(chunk)
        fingerprint=digest.hexdigest()
    report=dict(environment=dict(python=platform.python_version(),platform=platform.platform(),
                numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,opencv_threads=cv2.getNumThreads()),
                input=str(Path(args.input).resolve()),input_sha256=fingerprint,bbox=args.bbox,
                rotation=args.with_rotation,scale=not args.no_scale,backends={},status='incomplete',
                note='Stage sample uses a separate tracker; nested stage times must not be added together.')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(serializable(report),indent=2,allow_nan=False)+'\n')
    failed=False
    for name in names:
        entry=report['backends'][name]={'status':'running','runs':[]};save()
        try:
            warm=run_once(args,name,limit=args.warmup_frames)
            entry['cold_startup_seconds']=warm['startup_seconds']
            entry['warmup_initialization_seconds']=warm['initialization_seconds']
            entry['warmup_total_seconds']=warm['application_total_seconds']
            entry['warmup_frame_seconds']=warm['frame_seconds']
            entry['backend']=warm['backend']
            for number in range(args.repeats):
                output=args.application_output/name/str(number+1) if args.application_output else None
                run=run_once(args,name,limit=args.max_frames,output=output)
                entry['runs'].append(run);save()
                print(f'{name} run {number+1}: {run["frames"]} frames, {run["tracker_fps"]:.3f} tracker FPS',file=sys.stderr,flush=True)
            if args.profile_frames:
                sample_output=args.application_output/name/'stage-sample' if args.application_output else None
                entry['stage_sample']=run_once(args,name,
                    limit=min(args.profile_frames,args.max_frames) if args.max_frames else args.profile_frames,
                    profile=True,output=sample_output)
            entry['tracker_fps']=summary([r['tracker_fps'] for r in entry['runs']])
            entry['status']='complete' if all(r['complete_sequence'] for r in entry['runs']) else 'partial_sequence'
        except Exception as exc:
            entry.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            failed=True
            print(f'{name} benchmark failed: {exc}',file=sys.stderr,flush=True)
        save()
    cpu=report['backends'].get('cpu',{});gpu=report['backends'].get('cupy',{})
    if cpu.get('tracker_fps') and gpu.get('tracker_fps'):
        report['sequence_parity']=sequence_parity(cpu['runs'],gpu['runs'])
        if report['sequence_parity']['status']!='passed': failed=True
        report['median_tracker_speedup']=gpu['tracker_fps']['median']/cpu['tracker_fps']['median']
        report['paired_run_speedups']=[g['tracker_fps']/c['tracker_fps'] for c,g in zip(cpu['runs'],gpu['runs'])]
        report['performance_target_met']=(min(report['paired_run_speedups'])>=1.25 and args.repeats>=3
            and cpu['status']==gpu['status']=='complete' and report['sequence_parity']['status']=='passed')
    report['status']='failed' if failed else ('complete' if all(v['status']=='complete' for v in report['backends'].values()) else 'partial_sequence')
    report['gpu_validation']='not_run' if 'cupy' not in report['backends'] else report['backends']['cupy']['status']
    report['support_status']='experimental; tracking correctness requires tests/run_gpu_validation.py'
    save()
    print(f'Wrote {args.output}',file=sys.stderr)
    return 1 if failed else 0


if __name__=='__main__': sys.exit(main())
