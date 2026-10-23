"""Train and evaluate one-step observation value and discrete evidence gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from map_baseline.environment import Dataset, DEVELOPMENT, VALIDATION, TRAIN, SIMPLE, HARD, GRID, LEGAL, NEXT, RANGES
from .model import Gate, Gain, candidates, classify, gate_input, gate_update, state_input, choose_gain

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[42,43,44,45,46]
BUDGETS=[0,10,20,40,60]
COEFFICIENTS=[0,.0005,.001,.002]
READOUTS=['final','max_confidence','mean','gate']
CONFIG={'epochs':20,'batch_size':2048,'lr':.001,'weight_decay':.001,'starts_per_scene':128,
        'history_moves':4,'max_moves':5,'projection_seed':1729,'budgets':BUDGETS,
        'coefficients':COEFFICIENTS,'selection_cost_per_m':.001,'seeds':SEEDS}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


class Observations:
    def __init__(self,scenes,device='cuda'):
        self.device=torch.device(device)
        raw=Dataset(ROOT/'output/map/cache',scenes)
        sam=Dataset(ROOT/'output/ibe_map/cache',scenes)
        for key in ('probs','boxes','visible','names'):
            if not np.array_equal(getattr(raw,key),getattr(sam,key)):raise ValueError(f'Raw/SAM mismatch {key}')
        self.scenes=raw.scenes
        self.slots=raw.scene_to_slot
        self.fingerprints={'raw':raw.fingerprint,'sam':sam.fingerprint,'observation':raw.observation_signature}
        self.sam_signature=sam.observation_signature
        self.p=torch.as_tensor(raw.probs,device=self.device,dtype=torch.float32)
        self.labels=torch.as_tensor(raw.labels,device=self.device)
        self.sam_features=sam.features
        rng=np.random.default_rng(CONFIG['projection_seed'])
        projection=torch.as_tensor(rng.standard_normal((2048,64))/np.sqrt(2048),device=self.device,dtype=torch.float32)
        pieces=[]
        for features in (raw.features,sam.features):
            part=[]
            for lo in range(0,len(features),8192):
                f=torch.as_tensor(features[lo:lo+8192],device=self.device,dtype=torch.float32)
                f=F.layer_norm(f,(2048,))
                part.append(f@projection)
            pieces.append(torch.cat(part))
        boxes=torch.as_tensor(raw.boxes,device=self.device,dtype=torch.float32)/256
        positions=torch.as_tensor(np.tile(GRID,(len(self.scenes),1)),device=self.device,dtype=torch.float32)/100
        self.obs=torch.cat((*pieces,boxes,positions),1)
        self.legal=torch.as_tensor(LEGAL,device=self.device)
        self.next=torch.as_tensor(NEXT,device=self.device,dtype=torch.long)
        self.cost=torch.as_tensor(RANGES*10,device=self.device,dtype=torch.float32)
        self.correct=classify(self.p)==self.labels

    def ids(self,scenes):
        return torch.as_tensor(np.concatenate([self.slots[s]*605+np.arange(605) for s in scenes]),device=self.device)

    def following(self,ids):
        return ids[:,None]//605*605+self.next[ids%605]


def fit(model,x,y,mask=None):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    history=[]
    for epoch in range(CONFIG['epochs']):
        order=torch.randperm(len(x),device=x.device)
        numerator=denominator=0.
        for idx in order.split(CONFIG['batch_size']):
            loss=F.binary_cross_entropy_with_logits(model(x[idx]),y[idx].float(),reduction='none')
            valid=torch.ones_like(loss) if mask is None else mask[idx]
            value=(loss*valid).sum()/valid.sum()
            opt.zero_grad(set_to_none=True);value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            opt.step()
            numerator+=float((loss.detach()*valid).sum());denominator+=float(valid.sum())
        history.append({'epoch':epoch+1,'bce':numerator/denominator})
    model.eval()
    return history


@torch.no_grad()
def exploration_action(data,ids,t):
    valid=data.legal[ids%605,:24]
    random_scores=torch.rand(valid.shape,device=data.device).masked_fill(~valid,-1)
    random_action=random_scores.argmax(1)
    # Ascend first, then alternate x/y directions. This uses position only.
    axis=2 if t==0 else (t-1)%2
    coord=torch.as_tensor(GRID,device=data.device)[ids%605,axis]
    direction=torch.full_like(ids,5) if axis==2 else 2*axis+(coord>0).long()
    moves=torch.arange(24,device=data.device)
    scores=(moves[None,:]//4==direction[:,None]).float()*10+data.cost[None,:24]/10
    geometric=scores.masked_fill(~valid,-1).argmax(1)
    return torch.where(torch.rand(len(ids),device=data.device)<.5,random_action,geometric)


@torch.no_grad()
def histories(data,scenes,gate=None):
    starts=torch.cat([data.ids([s])[torch.randperm(605,device=data.device)[:CONFIG['starts_per_scene']]] for s in scenes])
    ids=starts.clone();mem_obs=data.obs[ids].clone();mem_p=data.p[ids].clone()
    visited=ids[:,None]
    gx=[];gy=[];sx=[];state_ids=[];state_history=[]
    for t in range(CONFIG['history_moves']):
        if gate is not None:
            sx.append(state_input(data.obs[ids],data.p[ids],mem_obs,mem_p))
            state_ids.append(ids.clone())
            state_history.append(F.pad(visited,(0,CONFIG['history_moves']-visited.shape[1]),value=-1))
        actions=exploration_action(data,ids,t)
        nxt=data.following(ids).gather(1,actions[:,None]).squeeze(1)
        repeated=(visited==nxt[:,None]).any(1)
        new_obs,new_p=data.obs[nxt],data.p[nxt]
        if gate is None:
            gx.append(gate_input(mem_obs,mem_p,new_obs,new_p))
            ps=candidates(mem_p,new_p)
            gy.append(classify(ps)==data.labels[ids,None])
            # Diverse causal memories, independent of truth and candidate scores.
            op=torch.randint(3,(len(ids),),device=data.device)
            op=torch.where((mem_p.sum(1)==0)&(new_p.sum(1)>0),0,op)
            op=torch.where((new_p.sum(1)==0)|repeated,1,op)
            os=torch.stack((new_obs,mem_obs,(new_obs+mem_obs)/2),1)
            row=torch.arange(len(ids),device=data.device)
            mem_obs,mem_p=os[row,op],ps[row,op]
        else:
            mem_obs,mem_p,_=gate_update(gate,mem_obs,mem_p,new_obs,new_p,repeated)
        ids=nxt;visited=torch.cat((visited,ids[:,None]),1)
    if gate is None:return torch.cat(gx),torch.cat(gy)
    return torch.cat(sx),torch.cat(state_ids),torch.cat(state_history)


@torch.no_grad()
def joint_targets(data,gate,x,ids,visited):
    targets=[]
    for lo in range(0,len(ids),1024):
        current=ids[lo:lo+1024];n=len(current)
        next_ids=data.following(current)
        repeated=(next_ids[:,:,None]==visited[lo:lo+n,None,:]).any(2)
        memory_obs=x[lo:lo+n,140:275].repeat_interleave(25,0)
        memory_p=x[lo:lo+n,275:280].repeat_interleave(25,0)
        _,p,_=gate_update(gate,memory_obs,memory_p,data.obs[next_ids.flatten()],data.p[next_ids.flatten()],repeated.flatten())
        targets.append((classify(p).reshape(n,25)==data.labels[current,None]))
    return torch.cat(targets)


def training_identity(data,scenes,seed):
    return {'config':CONFIG,'training_scenes':scenes,'seed':seed,'cache':data.fingerprints,
            'code':{p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('model.py'))}}


def train_models(data,scenes,seed,path):
    identity=training_identity(data,scenes,seed)
    manifest=path/'training.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['identity']!=identity:raise ValueError(f'Training identity changed: choose another output {path}')
        models={}
        for name,cls in [('gate',Gate),('gain_final',Gain),('gain_gate',lambda:Gain(True))]:
            checkpoint=path/(name+'.pt')
            if digest(checkpoint)!=saved['checkpoint_sha256'][name]:raise ValueError('Checkpoint changed')
            model=cls().to(data.device);model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=data.device));model.eval();models[name]=model
        return models
    path.mkdir(parents=True,exist_ok=True);seed_all(seed);start=time.time()
    gate=Gate().to(data.device)
    x,y=histories(data,scenes)
    gate_history=fit(gate,x,y);del x,y
    ids=data.ids(scenes)
    final=Gain().to(data.device)
    final_history=fit(final,state_input(data.obs[ids],data.p[ids]),data.correct[data.following(ids)],data.legal[ids%605])
    x,walk_ids,visited=histories(data,scenes,gate)
    initial=state_input(data.obs[ids],data.p[ids],data.obs[ids],data.p[ids])
    x=torch.cat((initial,x));walk_ids=torch.cat((ids,walk_ids))
    visited=torch.cat((F.pad(ids[:,None],(0,3),value=-1),visited))
    y=joint_targets(data,gate,x,walk_ids,visited)
    joint=Gain(True).to(data.device)
    joint_history=fit(joint,x,y,data.legal[walk_ids%605]);del x,y
    models={'gate':gate,'gain_final':final,'gain_gate':joint}
    hashes={}
    for name,model in models.items():
        torch.save(model.state_dict(),path/(name+'.pt'));hashes[name]=digest(path/(name+'.pt'))
    write(manifest,{'identity':identity,'checkpoint_sha256':hashes,
                    'loss':{'gate':gate_history,'gain_final':final_history,'gain_gate':joint_history},
                    'seconds':time.time()-start})
    print(f'Trained seed {seed}, scenes {len(scenes)}, {time.time()-start:.1f}s',flush=True)
    return models


@torch.no_grad()
def reference_q(data,model,history):
    values=[]
    for part in history.split(512):
        f=torch.as_tensor(data.sam_features[part.cpu().numpy()],device=data.device,dtype=torch.float32)
        boxes=data.obs[part,128:132]*256
        q,_=model.forward_sequence(f,boxes)
        values.append(q[:,-1])
    return torch.cat(values)


@torch.no_grad()
def rollout(data,scenes,models,controller,budget,coefficient=0,seed=42,reference=None):
    seed_all(seed)
    start=data.ids(scenes);ids=start.clone();n=len(ids)
    remaining=torch.full((n,),float(budget),device=data.device)
    mem_obs=data.obs[ids].clone();mem_p=data.p[ids].clone()
    best_p=mem_p.clone();sum_p=mem_p.clone();count=(mem_p.sum(1)>0).float()
    visited=ids[:,None];history=torch.full((n,6),-1,device=data.device,dtype=torch.long);history[:,0]=ids
    actions=torch.full((n,5),-1,device=data.device,dtype=torch.long)
    active=torch.ones(n,device=data.device,dtype=torch.bool)
    moves=torch.zeros(n,device=data.device,dtype=torch.long)
    operations=torch.zeros((n,3),device=data.device,dtype=torch.long)
    gate_choices=torch.full((n,5),-1,device=data.device,dtype=torch.long)
    reasons=torch.zeros(n,device=data.device,dtype=torch.long) # 1 policy stop, 2 budget, 3 horizon
    for t in range(CONFIG['max_moves']):
        rows=active.nonzero().flatten()
        if not len(rows):break
        cur=ids[rows];legal=data.legal[cur%605].clone()
        legal[:,:24] &= data.cost[None,:24]<=remaining[rows,None]+1e-6
        affordable=legal[:,:24].any(1)
        if controller.startswith('gain'):
            joint=controller=='gain_gate'
            state=state_input(data.obs[cur],data.p[cur],mem_obs[rows] if joint else None,mem_p[rows] if joint else None)
            a=choose_gain(models[controller](state),legal,remaining[rows],data.cost,coefficient)
        elif controller=='ibe_reference':
            q=reference_q(data,reference,history[rows,:t+1])
            a=q.masked_fill(~legal,-torch.inf).argmax(1)
        elif controller=='stationary':a=torch.full_like(cur,24)
        else:
            if controller=='random':scores=torch.rand((len(rows),24),device=data.device)
            elif controller=='sweep':
                axis=2 if t==0 else (t-1)%2
                coord=torch.as_tensor(GRID,device=data.device)[cur%605,axis]
                direction=torch.full_like(cur,5) if axis==2 else 2*axis+(coord>0).long()
                options=torch.arange(24,device=data.device)
                scores=(options[None,:]//4==direction[:,None]).float()*10+data.cost[None,:24]/10
            else:raise ValueError(controller)
            # All baselines use the same legal mask; no privileged future scores.
            a=scores.masked_fill(~legal[:,:24],-torch.inf).argmax(1)
            a=torch.where(data.p[cur].max(1).values>=.9,24,a)
        a=torch.where(affordable,a,24)
        stop=a==24
        reasons[rows[stop]]=torch.where(affordable[stop],1,2)
        active[rows[stop]]=False
        moving=rows[~stop];a=a[~stop];cur=cur[~stop]
        if not len(moving):continue
        nxt=data.following(cur).gather(1,a[:,None]).squeeze(1)
        actions[moving,t]=a;history[moving,t+1]=nxt
        repeated=(history[moving,:t+1]==nxt[:,None]).any(1)
        new_obs,new_p=data.obs[nxt],data.p[nxt]
        mo,mp,op=gate_update(models['gate'],mem_obs[moving],mem_p[moving],new_obs,new_p,repeated)
        mem_obs[moving]=mo;mem_p[moving]=mp
        operations[moving,op]+=1
        gate_choices[moving,t]=op
        novel=~repeated & (new_p.sum(1)>0)
        replace=novel & (new_p.max(1).values>best_p[moving].max(1).values)
        best_p[moving[replace]]=new_p[replace]
        sum_p[moving]+=new_p*novel[:,None];count[moving]+=novel
        ids[moving]=nxt;remaining[moving]-=data.cost[a];moves[moving]+=1
    reasons[active]=3
    probabilities=torch.stack((data.p[ids],best_p,sum_p/count.clamp_min(1)[:,None],mem_p),1)
    predictions=classify(probabilities)
    result={k:v.cpu().numpy() for k,v in {'starts':start,'history':history,'actions':actions,'predictions':predictions,
            'final_probabilities':probabilities,'distance_m':budget-remaining,'moves':moves,'stop_reason':reasons,
            'gate_operations':operations,'gate_choices':gate_choices,'labels':data.labels[start]}.items()}
    return result


def metrics(result,scenes,controller,budget,seed,split):
    out=[]
    for subset in ['all',*scenes]:
        mask=np.ones(len(result['starts']),bool) if subset=='all' else np.arange(len(result['starts']))//605==scenes.index(subset)
        truth=result['labels'][mask];pred=result['predictions'][mask]
        base=pred[:,0]==truth
        for j,readout in enumerate(READOUTS):
            correct=pred[:,j]==truth
            out.append({'seed':seed,'split':split,'scene':subset,'controller':controller,'budget_m':budget,'readout':readout,
                        'episodes':len(truth),'accuracy':float(correct.mean()),'distance_m':float(result['distance_m'][mask].mean()),
                        'moves':float(result['moves'][mask].mean()),'unknown_rate':float((pred[:,j]<0).mean()),
                        'rescue_rate':float((correct & ~base).mean()),'harm_rate':float((~correct & base).mean()),
                        'policy_stop_rate':float((result['stop_reason'][mask]==1).mean()),
                        'gate_operation_counts':result['gate_operations'][mask].sum(0).tolist()})
    return out


@torch.no_grad()
def calibration(data,scenes,models):
    ids=data.ids(scenes);result={}
    for name in ('gain_final','gain_gate'):
        joint=name=='gain_gate'
        x=state_input(data.obs[ids],data.p[ids],data.obs[ids] if joint else None,data.p[ids] if joint else None)
        if joint:
            y=joint_targets(data,models['gate'],x,ids,F.pad(ids[:,None],(0,3),value=-1))
        else:y=data.correct[data.following(ids)]
        expected=models[name](x).sigmoid();valid=data.legal[ids%605]
        delta=expected[:,:24]-expected[:,24:25];actual=y[:,:24].float()-y[:,24:25].float();mask=valid[:,:24]
        edges=[-1,-.5,-.2,-.05,0,.05,.2,.5,1.000001];bins=[]
        for low,high in zip(edges,edges[1:]):
            select=mask&(delta>=low)&(delta<high);n=int(select.sum())
            bins.append({'low':low,'high':high,'n':n,'predicted':float(delta[select].mean()) if n else None,
                         'observed':float(actual[select].mean()) if n else None})
        result[name]={'brier':float(((expected-y.float()).square()*valid).sum()/valid.sum()),'gain_bins':bins,
                      'scope':'All validation starting views; one-step outcomes, initial evidence only.'}
    return result


def development(out,device):
    data=Observations(TRAIN,device);rows=[];cals={}
    for seed in SEEDS:
        models=train_models(data,DEVELOPMENT,seed,out/'development'/f'seed_{seed}')
        cals[str(seed)]=calibration(data,VALIDATION,models)
        for controller in ('gain_final','gain_gate'):
            for cost in COEFFICIENTS:
                for budget in BUDGETS[1:]:
                    result=rollout(data,VALIDATION,models,controller,budget,cost,seed)
                    readout=0 if controller=='gain_final' else 3
                    acc=float((result['predictions'][:,readout]==result['labels']).mean());distance=float(result['distance_m'].mean())
                    rows.append({'seed':seed,'controller':controller,'coefficient':cost,'budget_m':budget,'accuracy':acc,
                                 'distance_m':distance,'utility':acc-.001*distance})
        write(out/'development_progress.json',{'rows':rows,'calibration':cals})
        print(f'Validation seed {seed} complete',flush=True)
    selected={};candidates_summary=[]
    for controller in ('gain_final','gain_gate'):
        options=[]
        for cost in COEFFICIENTS:
            chosen=[r for r in rows if r['controller']==controller and r['coefficient']==cost]
            row={'controller':controller,'coefficient':cost,**{k:float(np.mean([r[k] for r in chosen])) for k in ('utility','accuracy','distance_m')}}
            options.append(row);candidates_summary.append(row)
        best=max(options,key=lambda r:(r['utility'],-r['distance_m'],r['coefficient']))
        selected[controller]=best['coefficient']
    write(out/'selection.json',{'config':CONFIG,'training_scenes':DEVELOPMENT,'validation_scenes':VALIDATION,
          'selected':selected,'candidates':candidates_summary,'rows':rows,'calibration':cals,
          'code':training_identity(data,DEVELOPMENT,42)['code'],'test_data_loaded':False})
    print('Frozen validation selections: '+str(selected),flush=True)


def final(out,device):
    selection=json.loads((out/'selection.json').read_text())
    if selection['config']!=CONFIG or selection['code']!={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('model.py'))}:
        raise ValueError('Selection/config/code changed; rerun development in a new output directory')
    # Test observations are first loaded after the selection artifact exists.
    data=Observations(list(range(1,151)),device);all_rows=[];artifacts=[]
    for seed in SEEDS:
        models=train_models(data,TRAIN,seed,out/'final'/f'seed_{seed}')
        from ibe_map.model import make_model
        checkpoint=ROOT/'output/ibe_multiseed'/f'seed_{seed}/ibe_map/train/last.pt'
        ref=make_model('ibe_map').to(data.device)
        ck=torch.load(checkpoint,map_location=data.device,weights_only=True)
        config=ck['config']
        if (config.get('variant')!='ibe_map' or config.get('seed')!=seed or
            config.get('training_scenes')!=TRAIN or config.get('observation_signature')!=data.sam_signature):
            raise ValueError('Historical reference checkpoint identity mismatch')
        ref.load_state_dict(ck['state_dict']);ref.eval()
        for split,scenes in [('simple',SIMPLE),('hard',HARD)]:
            for controller in ('stationary','sweep','random','ibe_reference','gain_final','gain_gate'):
                for budget in BUDGETS:
                    result=rollout(data,scenes,models,controller,budget,selection['selected'].get(controller,0),seed,ref)
                    path=out/'rollouts'/f'{seed}_{split}_{controller}_{budget}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,**result)
                    all_rows.extend(metrics(result,scenes,controller,budget,seed,split))
                    artifacts.append({'path':str(path.relative_to(out)),'sha256':digest(path),'seed':seed,'split':split,
                                      'controller':controller,'budget_m':budget,'reference_sha256':digest(checkpoint) if controller=='ibe_reference' else None})
            print(f'Evaluated seed {seed} / {split}',flush=True)
        write(out/'results.json',{'config':CONFIG,'selection_sha256':digest(out/'selection.json'),'cache':data.fingerprints,
              'rows':all_rows,'artifacts':artifacts,'complete':seed==SEEDS[-1]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','development','final'],default='all')
    parser.add_argument('--out',type=Path,default=ROOT/'output/active_evidence')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage in ('all','development'):development(args.out,args.device)
    if args.stage in ('all','final'):final(args.out,args.device)


if __name__=='__main__':main()
