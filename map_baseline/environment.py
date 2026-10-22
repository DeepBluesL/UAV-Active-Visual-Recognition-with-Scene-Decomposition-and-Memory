"""Shared discrete environment for learned and hand-coded policies.

One decision = STOP at the current view, or one 10..40 m movement followed by
an observation. The fifth decision ends the episode *after* its movement.
Rewards use the resulting observation; STOP has zero distance. No inference
of the target class is passed to the policy (labels belong to the evaluator).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

CLASSES = ['Hatchback', 'Pickup', 'Sedan', 'Sportscar', 'SUV']
TRAIN = [base+i for base in (0,30,60,90,120) for i in range(1,21)]
SIMPLE = [base+i for base in (0,30,60,90,120) for i in range(21,26)]
HARD = [base+i for base in (0,30,60,90,120) for i in range(26,31)]
VALIDATION = [base+i for base in (0,30,60,90,120) for i in (19,20)]
DEVELOPMENT = [s for s in TRAIN if s not in VALIDATION]
GRID = np.array(list(itertools.product(range(-50,51,10),range(-50,51,10),range(-100,-59,10))),dtype=np.int16)
# 6 direction branches x 4 ranges, then one distinct STOP. z is NED.
DIRECTIONS = ((0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1))
ACTION_NAMES = [f"{'xyz'[axis]} {sign*r*10:+d} m" for axis,sign in DIRECTIONS for r in range(1,5)]+['STOP']
STOP = 24
RANGES = np.array(list(range(1,5))*6+[0], dtype=np.int16)
POSITION_LOOKUP = {tuple(p): i for i,p in enumerate(GRID)}
NEXT = np.empty((605,25),dtype=np.int32)
LEGAL = np.zeros((605,25),dtype=bool)
for i,p in enumerate(GRID):
    for action in range(25):
        q=p.copy()
        if action != STOP:
            axis,sign=DIRECTIONS[action//4]
            q[axis]+=sign*int(RANGES[action])*10
        j=POSITION_LOOKUP.get(tuple(q))
        NEXT[i,action]=i if j is None else j
        LEGAL[i,action]=j is not None


@dataclass(frozen=True)
class Protocol:
    max_decisions: int = 5
    gamma: float = 0.9
    terminal_reward: float = 0.5
    step_penalty: float = 0.1
    distance_penalty_per_unit: float = 0.02
    recognition_threshold: float = 0.2
    success_mode: str = 'top1_and_true_probability'
    heuristic_stop_threshold: float = 0.9

    def __post_init__(self):
        if self.max_decisions < 1 or not 0 <= self.gamma <= 1:
            raise ValueError('Invalid horizon or discount')
        if not 0 <= self.recognition_threshold <= 1:
            raise ValueError('Invalid recognition threshold')
        if self.success_mode not in ('top1_and_true_probability','true_probability'):
            raise ValueError('Unknown success predicate')

    def as_dict(self):
        return asdict(self)


def split_manifest():
    assert len(TRAIN)==100 and len(SIMPLE)==len(HARD)==25
    assert not(set(TRAIN)&set(SIMPLE) or set(TRAIN)&set(HARD) or set(SIMPLE)&set(HARD))
    assert sorted(TRAIN+SIMPLE+HARD)==list(range(1,151))
    return {'train':TRAIN,'simple':SIMPLE,'hard':HARD,'development_train':DEVELOPMENT,
            'development_validation':VALIDATION,
            'note':'v1 Table I typos resolved by the stated 20/5/5 contexts per class; scene00 excluded.'}


class Dataset:
    def __init__(self, cache: Path, scenes):
        self.scenes = sorted(set(scenes))
        self.scene_to_slot={s:i for i,s in enumerate(self.scenes)}
        arrays={k:[] for k in ('features','probs','boxes','visible','names')}
        cache_hash=hashlib.sha256()
        self.observation_signature=None
        for scene in self.scenes:
            path=Path(cache)/'scenes'/f'{scene:03d}.npz'
            meta=path.with_suffix('.json')
            metadata=json.loads(meta.read_text(encoding='utf8'))
            signature=metadata['signature']
            common={k:signature[k] for k in ('pretrained_sha256','classifier_sha256','preprocess')}
            common['dataset_path']=signature['source']['dataset_path']
            if self.observation_signature is None:
                self.observation_signature=common
            elif self.observation_signature!=common:
                raise ValueError(f'Mixed classifier/encoder/preprocess/dataset cache: {path}')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['npz_sha256']:
                raise ValueError(f'Cache content hash mismatch: {path}')
            with np.load(path,allow_pickle=False) as data:
                if not np.array_equal(data['positions'],GRID):
                    raise ValueError(f'Grid ordering mismatch: {path}')
                if data['features'].shape!=(605,2048) or data['probs'].shape!=(605,5):
                    raise ValueError(f'Cache shape mismatch: {path}')
                for k in arrays:
                    arrays[k].append(data[k].copy())
            # Metadata is an auditable chain to source data and both encoders.
            cache_hash.update(meta.read_bytes())
        for key,parts in arrays.items():
            setattr(self,key,np.concatenate(parts))
        if not np.isfinite(self.features).all() or not np.isfinite(self.probs).all():
            raise ValueError('Non-finite cache')
        block_names=['Hatchback','Pickup','Sportscar','Sedan','SUV']
        self.labels=np.repeat([CLASSES.index(block_names[(s-1)//30]) for s in self.scenes],605)
        self.predicted=self.probs.argmax(axis=1)
        self.correct=self.visible & (self.predicted==self.labels)
        self.confidence=self.probs.max(axis=1)
        self.true_probability=self.probs[np.arange(len(self.labels)),self.labels]
        self.fingerprint=cache_hash.hexdigest()

    def indices(self, scenes, stride=1):
        return np.concatenate([self.scene_to_slot[s]*605+np.arange(0,605,stride) for s in scenes])

    def transition(self, indices, actions, decision_index, protocol=Protocol()):
        indices=np.asarray(indices,dtype=np.int64)
        actions=np.asarray(actions,dtype=np.int64)
        local=indices%605
        if not LEGAL[local,actions].all():
            raise ValueError('Illegal boundary action; all policies must use the shared mask')
        following=indices//605*605+NEXT[local,actions]
        done=(actions==STOP)|(np.asarray(decision_index)>=protocol.max_decisions-1)
        success=self.visible[following] & (self.true_probability[following]>protocol.recognition_threshold)
        if protocol.success_mode=='top1_and_true_probability':
            success &= self.correct[following]
        reward=np.where(done,np.where(success,protocol.terminal_reward,-protocol.terminal_reward),-protocol.step_penalty)
        reward=reward-protocol.distance_penalty_per_unit*RANGES[actions]
        return following,reward.astype(np.float32),done,success


def heuristic_actions(data, indices, decision_index, policy, rng, visited=None, stop_mode='native', fixed_moves=3, stop_threshold=0.9):
    indices=np.asarray(indices)
    actions=np.empty(len(indices),dtype=np.int64)
    for row,idx in enumerate(indices):
        local=int(idx)%605
        if policy=='stationary':
            actions[row]=STOP
            continue
        stop = data.confidence[idx]>=stop_threshold if stop_mode=='native' else decision_index>=fixed_moves
        if stop:
            actions[row]=STOP
            continue
        options=np.flatnonzero(LEGAL[local,:STOP])
        if visited is not None:
            unvisited=[a for a in options if int(NEXT[local,a]) not in visited[row]]
            if unvisited:
                options=np.asarray(unvisited)
        if policy=='random':
            actions[row]=rng.choice(options)
            continue
        pos=GRID[local]
        axis_order=([2,0,1],[0,1,2],[1,0,2],[0,1,2])[decision_index%4]
        for axis in axis_order:
            sign=-1 if axis==2 else (1 if pos[axis]<=0 else -1)
            eligible=[a for a in options if DIRECTIONS[a//4]==(axis,sign)]
            if eligible:
                actions[row]=max(eligible,key=lambda a:RANGES[a])
                break
        else:
            actions[row]=options[0]
    return actions


def save_manifest(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(split_manifest(),indent=2),encoding='utf8')
