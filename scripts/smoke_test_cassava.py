"""Safe local smoke test for the published 5-class cassava MobileNetV3 checkpoint.

Usage: .venv-phase2/Scripts/python scripts/smoke_test_cassava.py \
  --checkpoint checkpoints/cassava_mobilenetv3_best.pth \
  --manifest manifests/cassava.csv --data-root C:/path/to/data/extracted \
  --output-dir reports/smoke_cassava
No accuracy is computed: this checkpoint was trained on the source dataset.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from PIL import Image
import timm
import torch

LABELS = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
          "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy"]
EXPECTED_SHA256 = '4043a480d7a48d636f4feefbe31df4c52eeadd8edb8404d31319eddd807ac4bb'


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def preprocess(path,device):
    with Image.open(path) as image:
        rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    resized=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
    crop=resized[16:240,16:240,:]/255.0
    normalized=(crop-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    return torch.from_numpy(normalized.transpose(2,0,1).copy()).unsqueeze(0).to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=sha256(args.checkpoint)
    if checkpoint_hash!=EXPECTED_SHA256:
        raise RuntimeError(f'Unexpected checkpoint SHA-256: {checkpoint_hash}')
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=timm.create_model('mobilenetv3_small_100',pretrained=False,num_classes=5)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state_dict=state['model_state'] if isinstance(state,dict) and 'model_state' in state else state
    model.load_state_dict(state_dict,strict=True)
    model.to(device).eval()
    with args.manifest.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    samples=[]
    for idx,label in enumerate(LABELS):
        sample=next(r for r in rows if r['split']=='train' and int(r['label_index'])==idx)
        samples.append(sample)
    predictions=[]
    with torch.inference_mode():
        for position,row in enumerate(samples):
            input_tensor=preprocess(args.data_root/row['image_path'],device)
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            logits=model(input_tensor)
            if device.type=='cuda': torch.cuda.synchronize()
            latency=(time.perf_counter()-start)*1000
            if tuple(logits.shape)!=(1,5): raise RuntimeError(f'Unexpected output shape {tuple(logits.shape)}')
            probabilities=torch.softmax(logits[0],dim=0)
            pred=int(torch.argmax(probabilities))
            predictions.append({'image_path':row['image_path'],'manifest_label':row['canonical_label'],
                'predicted_index':pred,'predicted_label':LABELS[pred],
                'confidence':float(probabilities[pred]),'inference_ms':round(latency,3),
                'input_shape':str(tuple(input_tensor.shape)),'output_shape':str(tuple(logits.shape))})
        benchmark_input=preprocess(args.data_root/samples[0]['image_path'],device)
        for _ in range(5): model(benchmark_input)
        benchmark_ms=[]
        for _ in range(20):
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            model(benchmark_input)
            if device.type=='cuda': torch.cuda.synchronize()
            benchmark_ms.append((time.perf_counter()-start)*1000)
    csv_path=args.output_dir/'predictions.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=predictions[0].keys()); writer.writeheader(); writer.writerows(predictions)
    log={'status':'passed','checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_hash,
         'architecture':'mobilenetv3_small_100','checkpoint_classes':LABELS,
         'torch_version':torch.__version__,'timm_version':timm.__version__,'opencv_version':cv2.__version__,
         'device':str(device),'gpu_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
         'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
         'sample_count':len(predictions),'input_shape':[1,3,224,224],'output_shape':[1,5],
         'latency_ms_all':[r['inference_ms'] for r in predictions],
         'latency_ms_median_after_warmup':statistics.median(r['inference_ms'] for r in predictions[1:]),
         'repeat_forward_benchmark':{'warmup_runs':5,'timed_runs':20,
             'median_ms':round(statistics.median(benchmark_ms),3),
             'p95_ms':round(sorted(benchmark_ms)[18],3),
             'scope':'same preprocessed tensor, forward only, 4 CPU threads'},
         'note':'Five train-source images used only for shape/inference smoke test; predictions are not independent accuracy metrics.'}
    (args.output_dir/'smoke_test.json').write_text(json.dumps(log,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(log,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
