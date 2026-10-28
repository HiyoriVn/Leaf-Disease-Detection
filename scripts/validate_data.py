"""Check label/image referential integrity in extracted leaf datasets.

Usage: python scripts/validate_data.py --data-root C:/path/to/data --output reports/data_validation.json
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decode-images', action='store_true', help='Fully decode every image; slower but checks pixel readability')
    args = parser.parse_args()
    roots = list((args.data_root / 'extracted').iterdir())
    results = {}
    for root in roots:
        if not root.is_dir():
            continue
        results[root.name] = {}
        if args.decode_images:
            images = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.webp','.bmp'}]
            errors = []
            for index, path in enumerate(images, 1):
                try:
                    with Image.open(path) as image:
                        image.load()
                except Exception as exc:
                    errors.append({'path': str(path.relative_to(root)), 'error': str(exc)})
                if index % 5000 == 0:
                    print(f'{root.name}: decoded {index}/{len(images)}', flush=True)
            results[root.name]['pixel_decode'] = {'checked': len(images), 'error_count': len(errors), 'errors': errors[:100]}
        if (root / 'train.csv').exists():
            image_dir = root / 'train_images'
            image_names = {p.name for p in image_dir.glob('*.jpg')}
            with (root / 'train.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            names = [row['image_id'] for row in rows]
            mapping = json.loads((root / 'label_num_to_disease_map.json').read_text(encoding='utf-8'))
            results[root.name].update({
                'kind': 'cassava', 'train_rows': len(rows), 'train_images': len(image_names),
                'labels': dict(collections.Counter(row['label'] for row in rows)), 'mapping': mapping,
                'missing_images': sorted(set(names) - image_names)[:100],
                'unlabeled_images': sorted(image_names - set(names))[:100],
                'duplicate_image_ids': sum(v-1 for v in collections.Counter(names).values() if v>1),
                'unknown_labels': sorted({row['label'] for row in rows} - set(mapping)),
                'test_images': len(list((root / 'test_images').glob('*.jpg'))),
                'train_tfrecords': len(list((root / 'train_tfrecords').glob('*.tfrec')))})
        elif list(root.rglob('RoCoLE-json.json')):
            annotations = json.loads(next(root.rglob('RoCoLE-json.json')).read_text(encoding='utf-8'))
            photos = {p.name for p in root.rglob('Photos/*.jpg')}
            names = [row.get('External ID') for row in annotations]
            states = collections.Counter()
            classes = collections.Counter()
            groups = collections.defaultdict(set)
            for row in annotations:
                label = row.get('Label') or {}
                classes[label.get('classification')] += 1
                leaves = label.get('Leaf') or []
                states[leaves[0].get('state') if leaves else None] += 1
                match = re.match(r'^(C\d+P\d+)', row.get('External ID') or '')
                if match:
                    groups[match.group(1)].add(label.get('classification'))
            disagreements = []
            for row in annotations:
                label = row.get('Label') or {}
                state = ((label.get('Leaf') or [{}])[0]).get('state')
                expected = 'healthy' if label.get('classification') == 'healthy' else 'unhealthy'
                if state != expected:
                    disagreements.append({'image': row.get('External ID'),
                                          'classification': label.get('classification'), 'state': state})
            results[root.name].update({'kind': 'coffee', 'annotation_rows': len(annotations),
                'photos': len(photos), 'classification_counts': dict(classes), 'state_counts': dict(states),
                'state_class_disagreements': len(disagreements), 'disagreement_examples': disagreements,
                'missing_photos': sorted(set(names) - photos)[:100],
                'unannotated_photos': sorted(photos - set(names))[:100],
                'duplicate_external_ids': sum(v-1 for v in collections.Counter(names).values() if v>1),
                'filename_plant_groups': len(groups), 'groups_with_both_classes': sum(len(v)>1 for v in groups.values())})
        else:
            classes = collections.Counter(p.parent.name for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
            results[root.name].update({'kind': 'folder_labels', 'class_counts': dict(classes), 'splits_present': any(p.name.lower() in {'train','val','valid','validation','test'} for p in root.rglob('*') if p.is_dir())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
