"""Create image manifests and review queues without copying or changing source images.

Usage: python scripts/prepare_manifests.py --config configs/dataset_config.yaml
Uses the Phase-1 duplicate inventory and hashes each source image once for the manifest.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import yaml

FIELDS = ['dataset','image_path','raw_label','canonical_label','label_index','group_id','split','sha256','source','exclusion_reason']
SPLITS = ['train','val','internal_test']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha_inventory(profile):
    out = {}
    for group in profile['cross_image_exact_duplicates']:
        for dataset_root, rel in group['files']:
            out[(dataset_root, rel.replace('\\','/'))] = group['sha256']
    return out


def base_row(dataset, root, path, label, label_index, data_root, known_hashes):
    rel_in_root = path.relative_to(root).as_posix()
    return {'dataset':dataset, 'image_path':path.relative_to(data_root).as_posix(),
            'raw_label':label, 'canonical_label':label, 'label_index':label_index,
            'group_id':'', 'split':'',
            'sha256':known_hashes.get((root.name,rel_in_root)) or digest(path),
            'source':root.name+'.zip', 'exclusion_reason':''}


def split_stratified(rows, fractions, seed):
    by_label = collections.defaultdict(list)
    for row in rows:
        if not row['exclusion_reason']:
            by_label[row['canonical_label']].append(row)
    for label, items in sorted(by_label.items()):
        random.Random(f'{seed}:{label}').shuffle(items)
        n = len(items)
        n_train = round(n * fractions['train'])
        n_val = round(n * fractions['val'])
        for i, row in enumerate(items):
            row['split'] = 'train' if i < n_train else 'val' if i < n_train+n_val else 'internal_test'
    for row in rows:
        if row['exclusion_reason'] and not row['split']:
            row['split'] = 'excluded'


class DSU:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x,x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a,b = self.find(a),self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def coffee_split(rows, fractions, seed):
    by_hash = collections.defaultdict(list)
    for row in rows: by_hash[row['sha256']].append(row)
    dsu = DSU()
    for row in rows: dsu.find(row['group_id'])
    for group in by_hash.values():
        for row in group[1:]: dsu.union(group[0]['group_id'], row['group_id'])
    for row in rows: row['group_id'] = dsu.find(row['group_id'])
    review = []
    for group in by_hash.values():
        if len(group) < 2: continue
        labels = {r['canonical_label'] for r in group}
        conflict = len(labels) > 1
        review.append({'sha256':group[0]['sha256'], 'images':'|'.join(r['image_path'] for r in group),
                       'labels':'|'.join(r['canonical_label'] for r in group), 'conflict':conflict})
        for r in (group if conflict else sorted(group,key=lambda r:r['image_path'])[1:]):
            r['exclusion_reason'] = 'conflicting_exact_duplicate' if conflict else 'exact_duplicate'
    assign_group_splits(rows, fractions, seed)
    return review


def assign_group_splits(rows, fractions, seed):
    by_group = collections.defaultdict(list)
    for row in rows:
        if not row['exclusion_reason']: by_group[row['group_id']].append(row)
    totals = collections.Counter(r['canonical_label'] for rs in by_group.values() for r in rs)
    targets = {s:{label:count*fractions[s] for label,count in totals.items()} for s in SPLITS}
    assigned = {s:collections.Counter() for s in SPLITS}
    counts = collections.Counter()
    items = list(by_group.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda pair: -sum(1/max(1,totals[r['canonical_label']]) for r in pair[1]))
    for key, group in items:
        hist = collections.Counter(r['canonical_label'] for r in group)
        def cost(split):
            class_cost = sum(((assigned[split][k]+v-targets[split][k])**2-(assigned[split][k]-targets[split][k])**2)/max(1,totals[k])**2 for k,v in hist.items())
            target_size=sum(totals.values())*fractions[split]
            size_cost=((counts[split]+len(group)-target_size)**2-(counts[split]-target_size)**2)/max(1,sum(totals.values()))**2
            return class_cost+size_cost
        chosen = min(SPLITS,key=cost)
        assigned[chosen].update(hist)
        counts[chosen] += len(group)
        for r in group: r['split'] = chosen
    for row in rows:
        if row['exclusion_reason']: row['split'] = 'excluded'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    repo = args.config.resolve().parents[1]
    data_root = Path(config['data_root']).resolve(strict=True)
    out = repo/config['manifest_dir']
    profile = json.loads((repo/config['source_profile']).read_text(encoding='utf-8'))
    known = sha_inventory(profile)
    fractions = config['split_fractions']; seed = config['seed']
    roots = {key:next(p for p in data_root.iterdir() if p.is_dir() and key in p.name.lower()) for key in ('rice','cassava','coffee')}
    summary = {}
    # Rice: folder label only; exact copies excluded from active manifest.
    root = roots['rice']; image_paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    labels = sorted({p.parent.name.lower() for p in image_paths})
    rice = []
    for i,p in enumerate(image_paths,1):
        r = base_row('rice',root,p,p.parent.name,labels.index(p.parent.name.lower()),data_root,known)
        r['canonical_label'] = p.parent.name.lower(); r['group_id'] = r['sha256']
        rice.append(r)
        if i%3000==0: print(f'rice hashed {i}/{len(image_paths)}',flush=True)
    by_hash = collections.defaultdict(list)
    for r in rice: by_hash[r['sha256']].append(r)
    for group in by_hash.values():
        for r in sorted(group,key=lambda r:r['image_path'])[1:]: r['exclusion_reason']='exact_duplicate'
    split_stratified(rice,fractions,seed); write_csv(out/'rice.csv',rice)
    summary['rice'] = dict(collections.Counter(r['split'] for r in rice))
    # Cassava: preserve original label index; one unlabeled test image stays outside metrics.
    root = roots['cassava']; mapping = json.loads((root/'label_num_to_disease_map.json').read_text(encoding='utf-8'))
    with (root/'train.csv').open(encoding='utf-8-sig',newline='') as f: pairs=list(csv.DictReader(f))
    cassava=[]
    for i,row in enumerate(pairs,1):
        p=root/'train_images'/row['image_id']
        r=base_row('cassava',root,p,mapping[row['label']],int(row['label']),data_root,known)
        r['group_id']=r['sha256']; cassava.append(r)
        if i%5000==0: print(f'cassava hashed {i}/{len(pairs)}',flush=True)
    by_hash=collections.defaultdict(list)
    for r in cassava: by_hash[r['sha256']].append(r)
    for group in by_hash.values():
        for r in sorted(group,key=lambda r:r['image_path'])[1:]: r['exclusion_reason']='exact_duplicate'
    split_stratified(cassava,fractions,seed)
    for p in sorted((root/'test_images').glob('*.jpg')):
        r=base_row('cassava',root,p,'','',data_root,known)
        r.update(group_id=r['sha256'],split='unlabeled',exclusion_reason='unlabeled_test_no_metrics')
        cassava.append(r)
    write_csv(out/'cassava.csv',cassava); summary['cassava']=dict(collections.Counter(r['split'] for r in cassava))
    # Coffee: plant identifiers and cross-plant duplicate images are inseparable.
    root=roots['coffee']; ann=json.loads(next(root.rglob('RoCoLE-json.json')).read_text(encoding='utf-8'))
    photos={p.name:p for p in root.rglob('Photos/*.jpg')}
    coffee_labels=['healthy','red_spider_mite','rust_level_1','rust_level_2','rust_level_3','rust_level_4']
    coffee=[]; disagreements=[]
    for i,row in enumerate(ann,1):
        name=row['External ID']; p=photos[name]
        label=row['Label']['classification']
        r=base_row('coffee',root,p,label,coffee_labels.index(label),data_root,known)
        match=re.match(r'^(C\d+P\d+)',name)
        r['group_id']=match.group(1) if match else name
        state=(row['Label'].get('Leaf') or [{}])[0].get('state')
        if state != ('healthy' if label=='healthy' else 'unhealthy'):
            r['exclusion_reason']='state_class_disagreement'
            disagreements.append({'image_path':r['image_path'],'classification':label,'leaf_state':state})
        coffee.append(r)
    duplicates=coffee_split(coffee,fractions,seed)
    write_csv(out/'coffee.csv',coffee)
    write_csv(out/'coffee_label_disputes.csv',disagreements,['image_path','classification','leaf_state'])
    write_csv(out/'coffee_exact_duplicates.csv',duplicates,['sha256','images','labels','conflict'])
    summary['coffee']=dict(collections.Counter(r['split'] for r in coffee))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
