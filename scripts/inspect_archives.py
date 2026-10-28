"""Inventory ZIP archives and optionally extract them with path/space safeguards.

Usage: python scripts/inspect_archives.py --data-root C:/path/to/data --output reports/archive_inventory.json
       python scripts/inspect_archives.py --data-root C:/path/to/data --output reports/archive_inventory.json --test-crc
       python scripts/inspect_archives.py --data-root C:/path/to/data --extract --output reports/archive_inventory.json
"""
import argparse
import collections
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile
import zlib


def unsafe_reason(info):
    name = info.filename.replace('\\', '/')
    path = PurePosixPath(name)
    if not name or name.startswith('/') or name.startswith('//') or ':' in name.split('/')[0]:
        return 'absolute or drive path'
    if any(part in ('..', '') for part in name.rstrip('/').split('/')):
        return 'traversal or empty component'
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode) or (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
        return 'link or special file'
    if info.flag_bits & 1:
        return 'encrypted'
    return None


def inspect(path, test_crc=False):
    result = {'archive': str(path), 'archive_bytes': path.stat().st_size,
              'format': 'ZIP' if zipfile.is_zipfile(path) else 'unsupported/corrupt'}
    if result['format'] != 'ZIP':
        return result
    with zipfile.ZipFile(path) as archive:
        files = [i for i in archive.infolist() if not i.is_dir()]
        directories = [i for i in archive.infolist() if i.is_dir()]
        unsafe = [{'path': i.filename, 'reason': unsafe_reason(i)} for i in archive.infolist() if unsafe_reason(i)]
        folded = collections.Counter(i.filename.replace('\\', '/').rstrip('/').casefold() for i in archive.infolist())
        duplicates = [k for k, v in folded.items() if v > 1]
        exts = collections.Counter(Path(i.filename).suffix.lower() or '[none]' for i in files)
        roots = collections.Counter(i.filename.replace('\\', '/').split('/')[0] for i in files)
        total = sum(i.file_size for i in files)
        result.update(files=len(files), directory_entries=len(directories), uncompressed_bytes=total,
                      compressed_entry_bytes=sum(i.compress_size for i in files),
                      ratio=round(total / max(1, path.stat().st_size), 2),
                      extensions=dict(exts.most_common()), roots=dict(roots.most_common(30)),
                      unsafe=unsafe[:100], unsafe_count=len(unsafe),
                      duplicate_paths=duplicates[:100], duplicate_count=len(duplicates),
                      largest=[{'path': i.filename, 'bytes': i.file_size} for i in sorted(files, key=lambda i: i.file_size, reverse=True)[:10]],
                      metadata_candidates=[i.filename for i in files if Path(i.filename).suffix.lower() in ('.csv','.json','.txt','.md','.xml','.yaml','.yml','.xlsx','.parquet')][:100],
                      sample_paths=[i.filename for i in files[:30]])
        if test_crc:
            try:
                result['crc_first_bad_file'] = archive.testzip()
                result['crc_test'] = 'passed' if result['crc_first_bad_file'] is None else 'failed'
            except Exception as exc:
                result['crc_test'] = f'error: {exc}'
    return result


def existing_matches(target, info):
    if target.stat().st_size != info.file_size:
        return False
    crc = 0
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            crc = zlib.crc32(block, crc)
    return crc == info.CRC


def extract(path, destination, inventory, resume=False):
    if inventory.get('unsafe_count') or inventory.get('duplicate_count') or inventory.get('format') != 'ZIP':
        raise RuntimeError('Archive has unsafe/duplicate paths or is not ZIP')
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise RuntimeError(f'Destination is not empty: {destination}')
    needed = inventory['uncompressed_bytes'] + max(1_073_741_824, inventory['uncompressed_bytes'] // 10)
    free = shutil.disk_usage(destination.parent).free
    if free < needed:
        raise RuntimeError(f'Insufficient disk: free={free}, needed={needed}')
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    bytes_written = 0
    resumed = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if unsafe_reason(info):
                    raise RuntimeError(f'Unsafe member: {info.filename}')
                target = destination.joinpath(*PurePosixPath(info.filename.replace('\\', '/')).parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError(f'Path escape: {info.filename}')
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if not resume or not existing_matches(target, info):
                        raise RuntimeError(f'Existing file differs or resume disabled: {target}')
                    resumed += 1
                    count += 1
                    bytes_written += info.file_size
                    continue
                with archive.open(info) as source, target.open('xb') as sink:
                    shutil.copyfileobj(source, sink, 1024 * 1024)
                if target.stat().st_size != info.file_size:
                    raise RuntimeError(f'Size mismatch: {info.filename}')
                count += 1
                bytes_written += info.file_size
    except Exception:
        # Preserve partial extraction for inspection/recovery.
        raise
    return {'destination': str(destination), 'files': count, 'bytes': bytes_written,
            'resumed_files': resumed, 'status': 'complete'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-crc', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Verify existing files by size and CRC, then continue')
    args = parser.parse_args()
    root = args.data_root.resolve(strict=True)
    free_before = shutil.disk_usage(root).free
    archives = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.zip','.rar','.7z','.tar','.gz'))
    results = []
    for path in archives:
        print(f'Inspecting {path.name}', flush=True)
        item = inspect(path, args.test_crc)
        if args.extract:
            dest = root / 'extracted' / path.stem
            dest.parent.mkdir(exist_ok=True)
            try:
                item['extraction'] = extract(path, dest, item, args.resume)
            except Exception as exc:
                item['extraction'] = {'destination': str(dest), 'status': 'failed', 'error': str(exc)}
        results.append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'data_root': str(root), 'free_bytes_before_run': free_before,
                                           'free_bytes': shutil.disk_usage(root).free,
                                           'archives': results}, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'name': path.name, 'files': item.get('files'), 'bytes': item.get('uncompressed_bytes'),
                          'unsafe': item.get('unsafe_count'), 'duplicate': item.get('duplicate_count'),
                          'extraction': item.get('extraction')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
