"""Train MAP with replay/target Q-learning, then compare on identical starts.

Commands: python -m map_baseline.experiment train|evaluate [--help].
The visual backbone and classification model are frozen and cached. This is an
auditable MAP-style reimplementation, not unavailable author training code.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .environment import (ACTION_NAMES,CLASSES,DEVELOPMENT,GRID,HARD,LEGAL,RANGES,
                          SIMPLE,STOP,TRAIN,VALIDATION,Dataset,Protocol,heuristic_actions,save_manifest)
from .model import MAPNetwork,masked_argmax
from .checkpoints import SnapshotSchedule

ROOT=Path(__file__).resolve().parents[1]


class Runner:
    def __init__(self,data,device='cuda'):
        self.data=data
        self.device=torch.device(device if device!='cuda' or torch.cuda.is_available() else 'cpu')
        self.features=torch.as_tensor(data.features,device=self.device)
        self.boxes=torch.as_tensor(data.boxes,device=self.device)
        self.legal=torch.as_tensor(LEGAL,device=self.device)

    def q(self,model,indices):
        ids=torch.as_tensor(indices,dtype=torch.long,device=self.device)
        return model(self.features[ids],self.boxes[ids]),ids

    @torch.inference_mode()
    def choose(self,model,indices,allow_stop=True):
        result=[]
        for start in range(0,len(indices),512):
            q,ids=self.q(model,indices[start:start+512])
            valid=self.legal[ids%605].clone()
            if not allow_stop:
                valid[:,STOP]=False
            result.append(masked_argmax(q,valid).cpu().numpy())
        return np.concatenate(result)


def rollout(runner,model,starts,policy,protocol,seed=42,mode='native',fixed_moves=3):
    """Vectorized identical-horizon rollouts; retained indices allow exact replay."""
    data=runner.data
    n=len(starts)
    current=np.asarray(starts).copy()
    initial=current.copy()
    active=np.ones(n,dtype=bool)
    returns=np.zeros(n,np.float64)
    distance=np.zeros(n,np.float64)
    decisions=np.zeros(n,np.int16)
    actions=np.full((n,protocol.max_decisions),-1,np.int16)
    observations=np.full((n,protocol.max_decisions+1),-1,np.int64)
    rewards=np.zeros((n,protocol.max_decisions),np.float32)
    observations[:,0]=current
    rng=np.random.default_rng(seed)
    visited=[{int(idx%605)} for idx in starts]
    for t in range(protocol.max_decisions):
        rows=np.flatnonzero(active)
        if not len(rows):
            break
        idx=current[rows]
        if policy=='map':
            if mode=='fixed_moves' and t>=fixed_moves:
                a=np.full(len(rows),STOP,dtype=np.int64)
            else:
                a=runner.choose(model,idx,allow_stop=mode!='fixed_moves')
        else:
            a=heuristic_actions(data,idx,t,policy,rng,[visited[i] for i in rows],mode,fixed_moves,
                                stop_threshold=protocol.heuristic_stop_threshold)
        nxt,reward,done,_=data.transition(idx,a,t,protocol)
        actions[rows,t]=a
        rewards[rows,t]=reward
        observations[rows,t+1]=nxt
        returns[rows]+=protocol.gamma**t*reward
        distance[rows]+=RANGES[a]*10
        decisions[rows]+=1
        current[rows]=nxt
        active[rows[done]]=False
        for row,pos in zip(rows,nxt):
            visited[row].add(int(pos%605))
    assert not active.any()
    success=data.visible[current]&(data.true_probability[current]>protocol.recognition_threshold)
    if protocol.success_mode=='top1_and_true_probability':
        success &= data.correct[current]
    return {'starts':initial,'final':current,'returns':returns,'distance':distance,'decisions':decisions,
            'actions':actions,'observations':observations,'rewards':rewards,
            'correct':data.correct[current],'initial_correct':data.correct[initial],'success':success}


def aggregate(result):
    a=result['actions']
    last=a[np.arange(len(a)),result['decisions']-1]
    return {'episodes':len(a),'accuracy':float(result['correct'].mean()),
            'initial_accuracy':float(result['initial_correct'].mean()),
            'return_mean':float(result['returns'].mean()),'distance_m':float(result['distance'].mean()),
            'decisions':float(result['decisions'].mean()),'success_rate':float(result['success'].mean()),
            'stop_rate':float((last==STOP).mean()),'budget_end_rate':float((last!=STOP).mean()),
            'wrong_to_correct':int((~result['initial_correct']&result['correct']).sum()),
            'correct_to_wrong':int((result['initial_correct']&~result['correct']).sum()),
            'wrong_stop_count':int(((last==STOP)&~result['correct']).sum())}


class Replay:
    def __init__(self,capacity):
        self.capacity=capacity
        self.pos=self.size=0
        self.ids=np.empty(capacity,np.int64)
        self.next_ids=np.empty(capacity,np.int64)
        self.actions=np.empty(capacity,np.int64)
        self.rewards=np.empty(capacity,np.float32)
        self.done=np.empty(capacity,bool)

    def add(self,ids,actions,rewards,next_ids,done):
        slots=(np.arange(len(ids))+self.pos)%self.capacity
        for name,values in [('ids',ids),('actions',actions),('rewards',rewards),('next_ids',next_ids),('done',done)]:
            getattr(self,name)[slots]=values
        self.pos=(self.pos+len(ids))%self.capacity
        self.size=min(self.capacity,self.size+len(ids))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng=np.random.default_rng(args.seed)
    protocol=Protocol(success_mode=args.success_mode)
    scenes=args.scenes or (DEVELOPMENT if args.phase=='development' else TRAIN)
    validate=VALIDATION if args.phase=='development' and not args.scenes else []
    data=Dataset(args.cache,sorted(set(scenes+validate)))
    runner=Runner(data,args.device)
    model=MAPNetwork().to(runner.device)
    target=MAPNetwork().to(runner.device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    replay=Replay(args.replay_capacity)
    allowed=data.indices(scenes)
    current=rng.choice(allowed,size=args.environments)
    decision=np.zeros(args.environments,dtype=np.int16)
    ep_return=np.zeros(args.environments)
    ep_distance=np.zeros(args.environments)
    completed=transitions=updates=0
    history=[]
    recent=[]
    started=time.perf_counter()
    next_log=args.log_every
    best=-float('inf')
    args.out.mkdir(parents=True,exist_ok=True)
    snapshots=SnapshotSchedule(args.out,getattr(args,'save_every',0),args.episodes)
    save_manifest(args.out/'splits.json')
    config={'phase':args.phase,'seed':args.seed,'requested_episodes':args.episodes,'training_scenes':scenes,
            'validation_scenes':validate,'protocol':protocol.as_dict(),'learning_rate':1e-4,
            'optimizer':'Adam','batch_size':args.batch_size,'environments':args.environments,
            'replay_capacity':args.replay_capacity,'learning_starts':args.learning_starts,
            'updates_per_vector_step':args.updates_per_step,'target_update_every':args.target_every,
            'epsilon':{'start':1.,'final':.05,'linear_decay_fraction':.7},'double_dqn':True,
            'loss':'smooth_l1','gradient_clip_norm':10.,'encoder':'frozen ImageNet ResNet50 V1',
            'network':'2048->512 image,4->32 bbox,544->512 state,V1+type7+range4',
            'normalization':'per-observation LayerNorm on 2048 feature; bbox /256',
            'checkpoint_selection':'best validation return' if validate else 'final requested training budget',
            'dataset_fingerprint':data.fingerprint,'cache_path':str(args.cache.resolve()),
            'observation_signature':data.observation_signature,
            'device':str(runner.device),'torch_version':str(torch.__version__),
            'gpu':torch.cuda.get_device_name() if runner.device.type=='cuda' else None}
    def checkpoint(path):
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'config':config,'completed_episodes':completed,'transitions':transitions,'updates':updates},path)
    snapshots.capture(completed,checkpoint)
    while completed<args.episodes:
        model.eval()
        epsilon=max(.05,1.-.95*completed/max(1,args.episodes*.7))
        a=runner.choose(model,current)
        explore=rng.random(args.environments)<epsilon
        # Random legal-action sampling includes STOP, unlike the comparison's
        # confidence-stopped random movement policy.
        random_scores=rng.random((args.environments,25))
        random_scores[~LEGAL[current%605]]=-1
        a[explore]=random_scores.argmax(1)[explore]
        following,reward,done,_=data.transition(current,a,decision,protocol)
        replay.add(current,a,reward,following,done)
        ep_return+=protocol.gamma**decision*reward
        ep_distance+=RANGES[a]*10
        decision+=1
        transitions+=len(a)
        for i in np.flatnonzero(done):
            recent.append((float(ep_return[i]),float(ep_distance[i]),bool(data.correct[following[i]])))
        if len(recent)>5000:
            recent=recent[-5000:]
        completed+=int(done.sum())
        current=following
        current[done]=rng.choice(allowed,size=int(done.sum()))
        decision[done]=0
        ep_return[done]=0
        ep_distance[done]=0
        last_loss=None
        if replay.size>=args.learning_starts:
            model.train()
            for _ in range(args.updates_per_step):
                sample=rng.integers(replay.size,size=args.batch_size)
                q,ids=runner.q(model,replay.ids[sample])
                action=torch.as_tensor(replay.actions[sample],device=runner.device)
                estimate=q.gather(1,action[:,None]).squeeze(1)
                with torch.no_grad():
                    next_q,next_ids=runner.q(model,replay.next_ids[sample])
                    next_action=masked_argmax(next_q,runner.legal[next_ids%605])
                    target_q,_=runner.q(target,replay.next_ids[sample])
                    bootstrap=target_q.gather(1,next_action[:,None]).squeeze(1)
                    r=torch.as_tensor(replay.rewards[sample],device=runner.device)
                    terminal=torch.as_tensor(replay.done[sample],device=runner.device)
                    expected=r+protocol.gamma*(~terminal)*bootstrap
                loss=nn.functional.smooth_l1_loss(estimate,expected)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),10.)
                optimizer.step()
                updates+=1
                last_loss=float(loss.detach())
                if updates%args.target_every==0:
                    target.load_state_dict(model.state_dict())
        snapshots.capture(completed,checkpoint)
        if completed>=next_log or completed>=args.episodes:
            model.eval()
            row={'episodes':completed,'transitions':transitions,'updates':updates,'epsilon':epsilon,
                 'loss':last_loss,'train_recent_return':float(np.mean([x[0] for x in recent])),
                 'train_recent_accuracy':float(np.mean([x[2] for x in recent])),
                 'elapsed_seconds':time.perf_counter()-started}
            if validate:
                # All 605 views in each of the ten held-out TRAIN contexts.
                validation=aggregate(rollout(runner,model,data.indices(validate),'map',protocol,args.seed))
                row.update({f'validation_{k}':v for k,v in validation.items()})
                if validation['return_mean']>best:
                    best=validation['return_mean']
                    checkpoint(args.out/'best.pt')
            history.append(row)
            checkpoint(args.out/'last.pt')
            summary={'config':config,'completed_episodes':completed,'transitions':transitions,'updates':updates,
                     'elapsed_seconds':time.perf_counter()-started,'history':history,
                     'best_validation_return':best if validate else None}
            (args.out/'training.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
            print(json.dumps(row),flush=True)
            next_log+=args.log_every
    return summary


def make_example(data,result,row,policy,dataset_path,selection_note):
    n=int(result['decisions'][row])
    steps=[]
    for t in range(n+1):
        idx=int(result['observations'][row,t])
        scene=data.scenes[idx//605]
        steps.append({'image':str((dataset_path/'multiview_imgs'/f'{scene:02d}'/str(data.names[idx])).resolve()),
                      'bbox':data.boxes[idx].tolist(),'position':GRID[idx%605].tolist(),
                      'predicted_label':CLASSES[int(data.predicted[idx])] if data.visible[idx] else 'Unknown',
                      'confidence':float(data.confidence[idx]),'probabilities':data.probs[idx].tolist(),
                      'action':ACTION_NAMES[int(result['actions'][row,t])] if t<n else 'END'})
    return {'scene':str(scene),'true_label':CLASSES[int(data.labels[idx])],'policy':policy,'steps':steps,
            'distance_m':float(result['distance'][row]),'initial_correct':bool(result['initial_correct'][row]),
            'final_correct':bool(result['correct'][row]),'selection_note':selection_note}


def evaluate(args):
    from demo.render_results import render
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    protocol=Protocol(**checkpoint['config']['protocol'])
    split_scenes={'train':TRAIN,'simple':SIMPLE,'hard':HARD}
    selected={key:split_scenes[key] for key in args.splits}
    data=Dataset(args.cache,sorted({s for group in selected.values() for s in group}))
    if checkpoint['config'].get('observation_signature')!=data.observation_signature:
        raise ValueError('Checkpoint and evaluation cache use different observation models or preprocessing')
    if args.data.resolve()!=Path(data.observation_signature['dataset_path']).resolve():
        raise ValueError('Visualization dataset path does not match the evaluated cache')
    runner=Runner(data,args.device)
    model=MAPNetwork().to(runner.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    args.out.mkdir(parents=True,exist_ok=True)
    save_manifest(args.out/'splits.json')
    metrics=[]
    examples=[]
    fields=['split','mode','policy','scene','start_x','start_y','start_z','initial_correct','final_correct',
            'return','distance_m','decisions','stop_reason','final_confidence','success']
    with (args.out/'evaluation.csv').open('w',newline='',encoding='utf8') as f, gzip.open(args.out/'trajectories.jsonl.gz','wt',encoding='utf8') as trace:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for split,scenes in selected.items():
            starts=data.indices(scenes)
            modes=['native','fixed_moves'] if split=='hard' else ['native']
            for mode in modes:
                policies=['stationary','sweep','random','map'] if mode=='native' else ['sweep','random','map']
                for policy in policies:
                    result=rollout(runner,model,starts,policy,protocol,args.seed,mode)
                    summary={'split':split if mode=='native' else 'hard_fixed_3_moves','policy':policy,**aggregate(result)}
                    # Contexts, rather than densely correlated frames, are the resampling unit.
                    per_scene=[]
                    for scene in scenes:
                        mask=starts//605==data.scene_to_slot[scene]
                        per_scene.append([float(result['correct'][mask].mean()),float(result['returns'][mask].mean())])
                    bootstrap_rng=np.random.default_rng(1729)
                    values=np.array(per_scene)
                    sampled=values[bootstrap_rng.integers(len(values),size=(2000,len(values)))].mean(1)
                    summary['accuracy_scene_bootstrap_95ci']=np.quantile(sampled[:,0],[.025,.975]).tolist()
                    summary['return_scene_bootstrap_95ci']=np.quantile(sampled[:,1],[.025,.975]).tolist()
                    metrics.append(summary)
                    print(json.dumps(summary),flush=True)
                    for row,idx in enumerate(starts):
                        scene=data.scenes[idx//605]
                        count=int(result['decisions'][row])
                        taken=result['actions'][row,:count].tolist()
                        reason='policy_stop' if taken[-1]==STOP else 'max_decisions_after_move'
                        if policy in ('sweep','random') and taken[-1]==STOP:
                            reason='confidence_threshold' if mode=='native' else 'fixed_movement_budget'
                        positions=GRID[result['observations'][row,:count+1]%605].tolist()
                        final=int(result['final'][row])
                        record={'split':split,'mode':mode,'policy':policy,'scene':scene,
                                'start_x':int(GRID[idx%605,0]),'start_y':int(GRID[idx%605,1]),'start_z':int(GRID[idx%605,2]),
                                'initial_correct':bool(result['initial_correct'][row]),'final_correct':bool(result['correct'][row]),
                                'return':float(result['returns'][row]),'distance_m':float(result['distance'][row]),
                                'decisions':count,'stop_reason':reason,'final_confidence':float(data.confidence[final]),
                                'success':bool(result['success'][row])}
                        writer.writerow(record)
                        observed=result['observations'][row,:count+1]
                        trace.write(json.dumps({**record,'positions':positions,'actions':taken,
                                                'rewards':result['rewards'][row,:count].tolist(),
                                                'predictions':[CLASSES[int(data.predicted[i])] if data.visible[i] else 'Unknown' for i in observed],
                                                'probabilities':data.probs[observed].tolist()},separators=(',',':'))+'\n')
                    if split=='hard' and mode=='native' and policy=='map':
                        successes=np.flatnonzero(~result['initial_correct']&result['correct'])
                        failures=np.flatnonzero(result['initial_correct']&~result['correct'])
                        for rows,note in [(successes,'Post-selected wrong-to-correct example; use full-grid metrics for performance.'),
                                          (failures,'Post-selected failure: correct initial observation became incorrect.')]:
                            if len(rows):
                                examples.append(make_example(data,result,int(rows[0]),policy,args.data,note))
    native={(m['split'],m['policy']):m for m in metrics}
    if 'train' in selected and 'hard' in selected:
        for policy in ('stationary','sweep','random','map'):
            native[('hard',policy)]['generalization_gap_return']=native[('train',policy)]['return_mean']-native[('hard',policy)]['return_mean']
    training_path=args.checkpoint.parent/'training.json'
    training=json.loads(training_path.read_text()) if training_path.exists() else checkpoint['config']
    result={'protocol':{**protocol.as_dict(),'initial_positions':'all 605 annotated viewpoints per scene, exactly matched across policies',
                        'action_space':'six directions x ranges1..4 plus STOP; 10m per unit; identical boundary masks',
                        'classification':'same frozen ResNet18 checkpoint for all policies',
                        'native_stopping':'MAP learned STOP; sweep/random original 0.90 confidence stopping',
                        'fixed_moves_diagnostic':'exactly 3 moves then STOP; ignores all confidence/learned stopping; distance still variable',
                        'decision_timing':'movement then recognition/reward; max5 decisions includes fifth movement, max6 observations',
                        'evaluation_seed':args.seed,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                        'cache_fingerprint':data.fingerprint},
            'training':training,'metrics':metrics,'examples':examples,'classes':CLASSES,
            'limitations':[
                'MAP-style reimplementation from UEVAVD v1; author code/weights were not available. This is not an exact numerical paper reproduction.',
                'Frozen ImageNet ResNet50 encoder and trained projection/dueling heads; encoder finetuning is not reproduced.',
                'Direction/range advantage centering and STOP handling are explicit implementation choices; the referenced MAP full text was not accessible.',
                'Recognition success is top1-correct AND true-class probability>0.2 by default. The paper score wording is ambiguous; true_probability-only is also configurable.',
                'The classifier remains the existing demo model; its training recipe is not the unavailable author classifier.',
                'MAP and heuristics share inputs available from the current view, legal moves, reward and max5 decisions; their native stopping policies differ by design.',
                'Fixed3-move diagnostics control decision count, not exact traveled meters; inspect accuracy together with distance.',
                'One MAP training seed and one random-policy evaluation seed; bootstrap intervals describe variation across contexts, not training-seed uncertainty.',
                'Ground-truth boxes are supplied; no SAM, GRU, online target tracker or collision-aware physical flight is implemented.',
                'Displayed trajectories are selected illustrations; full evaluation rows and every observed trajectory are saved.'
            ],'sources':[{'title':'UEVAVD arXiv v1','url':'https://arxiv.org/html/2411.04348v1'},
                         {'title':'Author dataset repository','url':'https://github.com/Leo000ooo/UEVAVD_dataset'},
                         {'title':'MAP 2019 reference','url':'https://doi.org/10.1109/TII.2019.2890849'}]}
    result_path=args.out/'results.json'
    result_path.write_text(json.dumps(result,indent=2),encoding='utf8')
    if examples:
        sequence={'method':'MAP-style learned direction/range/STOP','limitations':result['limitations'],
                  'classes':CLASSES,'metrics':native.get(('hard','map'),{}),'episodes':examples}
        (args.out/'examples.json').write_text(json.dumps(sequence,indent=2),encoding='utf8')
        render(args.out/'examples.json',args.out/'examples')
    # Keep matplotlib/NumPy's BLAS runtime separate from PyTorch's OpenMP
    # runtime on this Windows installation; do not suppress duplicate-runtime
    # checks with KMP_DUPLICATE_LIB_OK.
    subprocess.run([sys.executable,'-c',
                    'import sys; from map_baseline.report import render_report; render_report(sys.argv[1],sys.argv[2])',
                    str(result_path),str(args.out)],check=True,cwd=ROOT)
    print(f'Report: {args.out.resolve()/"report.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    training=sub.add_parser('train')
    training.add_argument('--phase',choices=['development','full'],default='development')
    training.add_argument('--scenes',nargs='+',type=int)
    training.add_argument('--episodes',type=int,default=30000)
    training.add_argument('--environments',type=int,default=64)
    training.add_argument('--batch-size',type=int,default=256)
    training.add_argument('--replay-capacity',type=int,default=100000)
    training.add_argument('--learning-starts',type=int,default=2048)
    training.add_argument('--updates-per-step',type=int,default=1)
    training.add_argument('--target-every',type=int,default=500)
    training.add_argument('--log-every',type=int,default=5000)
    training.add_argument('--save-every',type=int,default=0,
                          help='Save initial/intermediate/final snapshots; 0 disables diagnostic snapshots')
    training.add_argument('--success-mode',choices=['top1_and_true_probability','true_probability'],default='top1_and_true_probability')
    evaluating=sub.add_parser('evaluate')
    evaluating.add_argument('--checkpoint',type=Path,default=ROOT/'output/map/full/last.pt')
    evaluating.add_argument('--data',type=Path,default=ROOT/'dataset')
    evaluating.add_argument('--splits',nargs='+',choices=['train','simple','hard'],default=['train','simple','hard'])
    for command in (training,evaluating):
        command.add_argument('--cache',type=Path,default=ROOT/'output/map/cache')
        command.add_argument('--out',type=Path)
        command.add_argument('--seed',type=int,default=42)
        command.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.out is None:
        args.out=ROOT/'output/map'/args.phase if args.command=='train' else ROOT/'output/map'
    if args.command=='train':
        if args.episodes<1 or args.learning_starts<args.batch_size or args.replay_capacity<args.learning_starts:
            parser.error('Invalid training budget or replay sizes')
        train(args)
    else:
        evaluate(args)


if __name__=='__main__':
    main()
