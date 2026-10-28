"""Compute 64-bit pHash candidates for a manifest; review before excluding any image.

Usage: python scripts/find_near_duplicates.py --config configs/dataset_config.yaml --dataset rice
Optional --apply-groups keeps candidate-connected images in the same split.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from prepare_manifests import DSU, assign_group_splits, write_csv
import yaml


def phash(path):
    with Image.open(path) as image:
        tiny=np.asarray(ImageOps.exif_transpose(image).convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
    coeff=dctn(tiny,type=2,norm='ortho')[:8,:8]
    median=np.median(coeff.flatten()[1:])
    value=0
    for bit in coeff.flatten(): value=(value<<1)|int(bit>median)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dataset',choices=['rice','cassava','coffee'],required=True)
    p.add_argument('--apply-groups',action='store_true')
    args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    repo=args.config.resolve().parents[1]
    data_root=Path(cfg['data_root'])
    manifest=repo/cfg['manifest_dir']/f'{args.dataset}.csv'
    with manifest.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    opts=cfg.get(f'{args.dataset}_near_duplicate',cfg['rice_near_duplicate'])
    threshold=int(opts['max_hamming_distance'])
    hashes=[]
    cache=repo/cfg['manifest_dir']/f'{args.dataset}_phash.csv'
    cached={}
    if cache.exists():
        with cache.open(encoding='utf-8',newline='') as f:
            cached={(r['image_path'],r.get('sha256')):int(r['phash'],16) for r in csv.DictReader(f)}
    for i,row in enumerate(rows,1):
        h=cached.get((row['image_path'],row['sha256']))
        if h is None: h=phash(data_root/row['image_path'])
        hashes.append(h)
        if i%1000==0: print(f'{args.dataset}: pHash {i}/{len(rows)}',flush=True)
    write_csv(cache,[{'image_path':r['image_path'],'sha256':r['sha256'],'phash':f'{h:016x}'} for r,h in zip(rows,hashes)],['image_path','sha256','phash'])
    pairs=[]; edges=[]; group_edges=[]
    for i,first in enumerate(rows):
        for j in range(i+1,len(rows)):
            if first['sha256']==rows[j]['sha256']: continue
            distance=(hashes[i]^hashes[j]).bit_count()
            if distance<=threshold:
                edges.append((i,j))
                if distance<=opts.get('group_max_hamming_distance',threshold): group_edges.append((i,j))
                if len(pairs)<opts['max_pairs_reported']:
                    pairs.append({'image_a':first['image_path'],'image_b':rows[j]['image_path'],
                                  'label_a':first['canonical_label'],'label_b':rows[j]['canonical_label'],
                                  'distance':distance,'sha256_equal':False})
        if i%1000==0: print(f'{args.dataset}: compared {i}/{len(rows)}',flush=True)
    out=repo/cfg['manifest_dir']
    write_csv(out/f'{args.dataset}_near_duplicates.csv',pairs,
              ['image_a','image_b','label_a','label_b','distance','sha256_equal'])
    priority=sorted(pairs,key=lambda row:(int(row['distance']),row['label_a']==row['label_b']))[:50]
    write_csv(out/f'{args.dataset}_near_duplicates_priority.csv',priority,
              ['image_a','image_b','label_a','label_b','distance','sha256_equal'])
    summary={'dataset':args.dataset,'method':'64-bit DCT pHash','max_distance':threshold,
             'candidate_pairs':len(edges),'rows_written':len(pairs),
             'group_max_distance':opts.get('group_max_hamming_distance',threshold),
             'grouped_pairs':len(group_edges),
             'cross_label_pairs':sum(rows[i]['canonical_label']!=rows[j]['canonical_label'] for i,j in edges),
             'action':'review_only'}
    if args.apply_groups:
        dsu=DSU()
        for row in rows: dsu.find(row['group_id'])
        for i,j in group_edges: dsu.union(rows[i]['group_id'],rows[j]['group_id'])
        for row in rows: row['group_id']=dsu.find(row['group_id'])
        assign_group_splits(rows,cfg['split_fractions'],cfg['seed'])
        write_csv(manifest,rows)
        summary['action']='candidate_connected_groups_kept_in_same_split; no exclusions added'
        summary['split_counts']=dict(collections.Counter(r['split'] for r in rows))
        totals=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        totals[args.dataset]=summary['split_counts']
        (out/'summary.json').write_text(json.dumps(totals,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/f'{args.dataset}_near_duplicates_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
