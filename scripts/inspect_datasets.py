"""Profile extracted leaf datasets without loading image corpora into memory.

Usage: python scripts/inspect_datasets.py --data-root C:/path/to/data --output reports/dataset_profile.json
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def profile_folder(root, global_hashes):
    files = [p for p in root.rglob('*') if p.is_file()]
    dirs = sum(p.is_dir() for p in root.rglob('*'))
    result = {'root': str(root), 'files': len(files), 'directories': dirs,
              'bytes': sum(p.stat().st_size for p in files),
              'extensions': dict(collections.Counter(p.suffix.lower() or '[none]' for p in files)),
              'top_level': dict(collections.Counter(p.relative_to(root).parts[0] for p in files)),
              'metadata_files': [str(p.relative_to(root)) for p in files if p.suffix.lower() in {'.csv','.json','.xlsx','.md','.txt','.xml'}][:100]}
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    dimensions = collections.Counter()
    formats = collections.Counter()
    labels_by_parent = collections.Counter()
    errors = []
    for number, path in enumerate(images, 1):
        rel = str(path.relative_to(root))
        try:
            with Image.open(path) as image:
                dimensions[f'{image.width}x{image.height}'] += 1
                formats[image.format or 'unknown'] += 1
            digest = file_digest(path)
            global_hashes[digest].append((root.name, rel))
            labels_by_parent[path.parent.name] += 1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            errors.append({'path': rel, 'error': str(exc)})
        if number % 5000 == 0:
            print(f'{root.name}: {number}/{len(images)} image headers and SHA-256', flush=True)
    result['images'] = {'count': len(images), 'formats': dict(formats),
                        'dimensions': dict(dimensions.most_common(30)),
                        'distinct_dimensions': len(dimensions),
                        'parent_counts': dict(labels_by_parent),
                        'header_or_read_errors': errors[:100], 'error_count': len(errors),
                        'check_scope': 'all image headers and full image-byte SHA-256; no full pixel decode'}
    return result


def csv_profile(path, key=None, target=None):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = 0
        missing = collections.Counter()
        target_counts = collections.Counter()
        key_counts = collections.Counter()
        seen_rows = set()
        duplicate_rows = 0
        for row in reader:
            rows += 1
            missing.update(k for k, v in row.items() if v is None or v == '')
            if target:
                target_counts[row.get(target)] += 1
            if key:
                key_counts[row.get(key)] += 1
            signature = tuple(row.items())
            if signature in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(signature)
        return {'path': str(path), 'columns': reader.fieldnames, 'rows': rows,
                'missing': dict(missing), 'target_counts': dict(target_counts),
                'duplicate_keys': sum(v - 1 for v in key_counts.values() if v > 1),
                'duplicate_rows': duplicate_rows, 'keys': list(key_counts) if key else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = sorted(p for p in (args.data_root / 'extracted').iterdir() if p.is_dir())
    global_hashes = collections.defaultdict(list)
    result = {'datasets': [], 'cross_image_exact_duplicates': []}
    for root in roots:
        print(f'Profiling {root.name}', flush=True)
        item = profile_folder(root, global_hashes)
        csv_files = list(root.rglob('*.csv'))
        item['csv'] = []
        for path in csv_files:
            if path.name == 'train.csv':
                item['csv'].append(csv_profile(path, 'image_id', 'label'))
            elif path.name == 'RoCoLE-csv.csv':
                item['csv'].append(csv_profile(path, 'External ID'))
            else:
                item['csv'].append(csv_profile(path))
        result['datasets'].append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['cross_image_exact_duplicates'] = [{'sha256': digest, 'files': locs} for digest, locs in global_hashes.items() if len(locs) > 1]
    result['duplicate_summary'] = {'groups': len(result['cross_image_exact_duplicates']),
                                   'excess_images': sum(len(g['files']) - 1 for g in result['cross_image_exact_duplicates'])}
    for root in roots:
        annotation_files = list(root.rglob('RoCoLE-json.json'))
        if not annotation_files:
            continue
        rows = json.loads(annotation_files[0].read_text(encoding='utf-8'))
        labels = {row.get('External ID'): (row.get('Label') or {}).get('classification') for row in rows}
        groups = [g for g in result['cross_image_exact_duplicates']
                  if all(loc[0] == root.name for loc in g['files'])]
        result['duplicate_summary'][root.name] = {
            'groups': len(groups),
            'different_class_groups': sum(len({labels.get(Path(loc[1]).name) for loc in g['files']}) > 1 for g in groups),
            'different_binary_state_groups': sum(len({'healthy' if labels.get(Path(loc[1]).name) == 'healthy' else 'unhealthy' for loc in g['files']}) > 1 for g in groups)}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Exact duplicate groups: {len(result['cross_image_exact_duplicates'])}", flush=True)


if __name__ == '__main__':
    main()
