"""Resumeable SAM decomposition and frozen ResNet50 feature cache.

Classifier scores are copied byte-for-byte from the existing raw-view cache;
SAM changes only the policy's visual input, never the recognition evaluator.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from map_baseline.features import (MEAN, STD, ResNet50, read_scene, sha256,
                                   valid_cache, validate_arrays, write_json)
from .sam import ROOT, SAM_SETTINGS, SAMDecomposer


def load_rgb(record):
    with Image.open(record['path']) as image:
        return np.asarray(image.convert('RGB')).copy()


def encoder_tensor(image):
    image = Image.fromarray(image).resize((224,224), Image.Resampling.BILINEAR)
    x = np.asarray(image,dtype=np.float32).transpose(2,0,1)/255
    return torch.from_numpy((x-MEAN)/STD)


def save_simplified(item):
    path, image = item
    Image.fromarray(image).save(path, compress_level=1)


def save_preview(rgb, result, box, path):
    """Diagnostic images from actual segmentation, no synthetic observations."""
    panels = [Image.fromarray(rgb), Image.fromarray(result['target_mask'].astype(np.uint8)*255).convert('RGB'),
              Image.fromarray(result['simplified'])]
    draw = ImageDraw.Draw(panels[0])
    if box[2]>box[0] and box[3]>box[1]:
        draw.rectangle(tuple(box),outline='red',width=2)
    for x,y in result['points']:
        draw.ellipse((x-3,y-3,x+3,y+3),fill='yellow')
    canvas = Image.new('RGB',(768,280),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(panel,title) in enumerate(zip(panels,['Input + bbox / ring points','SAM target mask','SAM simplified policy input'])):
        canvas.paste(panel,(i*256,24)); draw.text((i*256+4,5),title,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'dataset')
    p.add_argument('--raw-cache',type=Path,default=ROOT/'output/map/cache')
    p.add_argument('--out',type=Path,default=ROOT/'output/ibe_map/cache')
    p.add_argument('--images',type=Path,default=ROOT/'output/ibe_map/decomposed')
    p.add_argument('--scenes',type=int,nargs='+',default=list(range(1,151)))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cuda')
    p.add_argument('--benchmark',type=int,default=0,help='Only time N images of the first scene; no production cache')
    args=p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    sam_weights=ROOT/'output/ibe_map/weights/sam_vit_b_01ec64.pth'
    sam_sha=sha256(sam_weights)
    source_manifest=json.loads((ROOT/'third_party/segment_anything/SOURCE.json').read_text())
    for filename,expected in source_manifest['files_sha256'].items():
        if sha256(ROOT/'third_party/segment_anything'/filename)!=expected:
            raise ValueError(f'SAM upstream source content changed: {filename}')
    sam_signature={**SAM_SETTINGS,'checkpoint_sha256':sam_sha,'source_commit':source_manifest['commit'],
                   'decomposer_code_sha256':sha256(ROOT/'ibe_map/sam.py'),
                   'attention_code_sha256':sha256(ROOT/'ibe_map/fast_attention.py'),
                   'decoder_code_sha256':sha256(ROOT/'ibe_map/batched_decoder.py'),
                   'cache_pipeline_code_sha256':sha256(Path(__file__))}
    sam=None
    encoder=ResNet50()
    weights=args.raw_cache/'weights/resnet50-0676ba61.pth'
    encoder_sha=sha256(weights)
    encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    encoder.to(device).eval().requires_grad_(False)
    args.out.mkdir(parents=True,exist_ok=True)
    scene_out=args.out/'scenes'; scene_out.mkdir(exist_ok=True)
    started=time.perf_counter(); extracted=[]; skipped=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for scene in sorted(set(args.scenes)):
            records,source=read_scene(args.data,scene)
            raw_path=args.raw_cache/'scenes'/f'{scene:03d}.npz'
            raw_meta=json.loads(raw_path.with_suffix('.json').read_text())
            if raw_meta['signature']['pretrained_sha256'] != encoder_sha:
                raise ValueError(f'ResNet50 weights differ from original cache identity: scene{scene}')
            if raw_meta['signature']['source'] != source or sha256(raw_path)!=raw_meta['npz_sha256']:
                raise ValueError(f'Original observation cache outdated or corrupt: scene{scene}')
            preprocess={**raw_meta['signature']['preprocess'],'scene_decomposition':sam_signature,
                        'features':'global average pooled2048 of actual SAM simplified image, float16',
                        'classifier_input':'unchanged ORIGINAL-view probabilities, copied from raw cache'}
            signature={**raw_meta['signature'],'preprocess':preprocess}
            npz_path=scene_out/f'{scene:03d}.npz'; meta_path=npz_path.with_suffix('.json')
            if not args.benchmark and valid_cache(npz_path,meta_path,signature):
                skipped.append(scene)
                print(f'scene{scene:03d}: valid cache reused',flush=True)
                continue
            scene_start=time.perf_counter()
            if sam is None:
                sam=SAMDecomposer(sam_weights,str(device))
            if args.benchmark:
                images=list(pool.map(load_rgb,records[:args.benchmark]))
                # Warmup also verifies real bbox segmentation before timing.
                sam.decompose_batch(images[:1],[records[0]['box']])
                if device.type=='cuda': torch.cuda.synchronize()
                t=time.perf_counter()
                for k in range(0,len(images),args.batch_size):
                    results=sam.decompose_batch(images[k:k+args.batch_size],[r['box'] for r in records[k:k+args.batch_size]])
                    if k==0:
                        for j,result in enumerate(results[:3]):
                            save_preview(images[j],result,records[j]['box'],args.out.parent/'sam_preview'/f'benchmark_{j}.png')
                if device.type=='cuda': torch.cuda.synchronize()
                elapsed=time.perf_counter()-t
                print(json.dumps({'images':len(images),'seconds':elapsed,'seconds_per_image':elapsed/len(images),
                                  'estimated_full_minutes':elapsed/len(images)*90750/60,
                                  'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None}),flush=True)
                return
            image_dir=args.images/f'{scene:03d}'; image_dir.mkdir(parents=True,exist_ok=True)
            progress_path=image_dir/'progress.json'
            progress={'signature':signature,'completed':0}
            if progress_path.exists():
                previous=json.loads(progress_path.read_text())
                if previous.get('signature')==signature:
                    progress=previous
            completed=progress['completed']
            for k in range(completed,len(records),args.batch_size):
                subset=records[k:k+args.batch_size]
                images=list(pool.map(load_rgb,subset))
                results=sam.decompose_batch(images,[r['box'] for r in subset])
                list(pool.map(save_simplified,[(image_dir/r['name'],v['simplified']) for r,v in zip(subset,results)]))
                for offset,(record,image,result) in enumerate(zip(subset,images,results)):
                    if k+offset in (0,300,604) and scene in (1,19,21,26,31,61,91,121):
                        preview=args.out.parent/'sam_preview'/f'{scene:03d}_{k+offset:03d}.png'
                        save_preview(image,result,record['box'],preview)
                        np.savez_compressed(preview.with_suffix('.npz'),**{key:value for key,value in result.items() if key!='simplified'})
                progress['completed']=k+len(subset)
                write_json(progress_path,progress)
                if progress['completed']%128==0 or progress['completed']==len(records):
                    print(f'scene{scene:03d} SAM {progress["completed"]}/605 elapsed{time.perf_counter()-scene_start:.1f}s',flush=True)
            features=[]
            for k in range(0,len(records),128):
                paths=[{'path':image_dir/r['name']} for r in records[k:k+128]]
                images=list(pool.map(load_rgb,paths))
                batch=torch.stack([encoder_tensor(image) for image in images]).to(device)
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    features.append(encoder(batch).float().cpu().numpy().astype(np.float16))
            with np.load(raw_path,allow_pickle=False) as original:
                arrays={k:original[k].copy() for k in original.files}
            arrays['features']=np.concatenate(features)
            validate_arrays(arrays)
            temporary=npz_path.with_suffix('.npz.tmp')
            with temporary.open('wb') as f: np.savez_compressed(f,**arrays)
            temporary.replace(npz_path)
            write_json(meta_path,{'scene':scene,'signature':signature,'npz_sha256':sha256(npz_path),
                                 'image_count':605,'visible_count':int(arrays['visible'].sum()),
                                 'raw_npz_sha256':raw_meta['npz_sha256'],'simplified_images':str(image_dir.resolve()),
                                 'extract_seconds':time.perf_counter()-scene_start})
            extracted.append(scene)
            print(f'completed scene{scene:03d}; new{len(extracted)} reused{len(skipped)}; total{time.perf_counter()-started:.1f}s',flush=True)
    write_json(args.out/'cache_manifest.json',{'scene_decomposition':sam_signature,'requested_scenes':args.scenes,
               'extracted':extracted,'reused':skipped,'elapsed_seconds':time.perf_counter()-started,
               'classifier_unchanged':True,'sam_finetuned':False})


if __name__=='__main__':
    main()
