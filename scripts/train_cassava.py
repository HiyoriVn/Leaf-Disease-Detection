"""Train a configured timm ImageNet backbone on frozen Cassava splits.

The internal-test manifest rows are checked for split integrity, but their
images are not decoded or scored. Select the best validation checkpoint
first, then run evaluate_cassava_new.py once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
import timm
import torch
import safetensors
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.cassava_inference import EXPECTED_LABELS
from src.classification_metrics import classification_metrics


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path, data_root):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    usable = [r for r in rows if r["split"] in ("train", "val", "internal_test") and not r["exclusion_reason"]]
    groups = {}
    hashes = {}
    for row in usable:
        label = int(row["label_index"])
        if label not in range(5) or row["canonical_label"] != EXPECTED_LABELS[label]:
            raise ValueError(f"Cassava label contract mismatch: {row['image_path']}")
        if not (data_root / row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        for mapping, key in ((groups, "group_id"), (hashes, "sha256")):
            prior = mapping.setdefault(row[key], row["split"])
            if prior != row["split"]:
                raise ValueError(f"{key} crosses splits")
    counts = {split: sum(r["split"] == split for r in usable) for split in ("train", "val", "internal_test")}
    if counts != {"train": 14978, "val": 3210, "internal_test": 3209}:
        raise ValueError(f"frozen Cassava split counts changed: {counts}")
    return usable, counts


def balanced_pilot(rows, per_class, seed):
    rng = random.Random(seed)
    return [row for i in range(5) for row in rng.sample([r for r in rows if int(r["label_index"]) == i], per_class)]


class CassavaDataset(Dataset):
    def __init__(self, rows, data_root, augment, seed, resize=256, crop=224,
                 interpolation=cv2.INTER_AREA, resize_mode="square",
                 augmentation_profile="legacy", train_crop="center"):
        self.rows = rows
        self.data_root = data_root
        self.augment = augment
        self.seed = seed
        self.resize = resize
        self.crop = crop
        self.interpolation = interpolation
        self.resize_mode = resize_mode
        self.augmentation_profile = augmentation_profile
        self.train_crop = train_crop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data_root / row["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        if self.resize_mode == "square":
            resized = cv2.resize(rgb, (self.resize, self.resize), interpolation=self.interpolation)
        elif self.resize_mode == "shortest_edge":
            height, width = rgb.shape[:2]
            scale = self.resize / min(height, width)
            out_w, out_h = max(self.resize, round(width * scale)), max(self.resize, round(height * scale))
            resized = cv2.resize(rgb, (out_w, out_h), interpolation=self.interpolation)
        else:
            raise ValueError(f"unknown resize_mode: {self.resize_mode}")
        if self.augment:
            if self.augmentation_profile == "legacy":
                if random.random() < 0.5:
                    resized = np.flip(resized, axis=1)
                if random.random() < 0.5:
                    resized = np.flip(resized, axis=0)
                resized = np.rot90(resized, random.randrange(4))
                contrast = random.uniform(0.9, 1.1)
                brightness = random.uniform(0.9, 1.1)
                resized = np.clip(((resized - 127.5) * contrast + 127.5) * brightness, 0, 255)
            elif self.augmentation_profile == "symptom_preserving":
                # Preserve leaf orientation semantics; use only mild geometric
                # and photometric changes to avoid erasing small lesions.
                if random.random() < 0.5:
                    resized = np.flip(resized, axis=1)
                angle = random.uniform(-10.0, 10.0)
                height, width = resized.shape[:2]
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                resized = cv2.warpAffine(resized, matrix, (width, height),
                                         flags=self.interpolation, borderMode=cv2.BORDER_REFLECT_101)
                contrast = random.uniform(0.94, 1.06)
                brightness = random.uniform(-8.0, 8.0)
                resized = np.clip((resized - 127.5) * contrast + 127.5 + brightness, 0, 255)
            elif self.augmentation_profile != "none":
                raise ValueError(f"unknown augmentation_profile: {self.augmentation_profile}")
        height, width = resized.shape[:2]
        if self.augment and self.train_crop == "random":
            top = random.randint(0, height - self.crop)
            left = random.randint(0, width - self.crop)
        else:
            top, left = (height - self.crop) // 2, (width - self.crop) // 2
        crop = resized[top:top + self.crop, left:left + self.crop] / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        tensor = torch.from_numpy(((crop - mean) / std).transpose(2, 0, 1).copy())
        return tensor, int(row["label_index"])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_trainable(model, stage):
    if stage == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    head_modules = [getattr(model, name, None) for name in ("conv_head", "classifier", "head")]
    for module in head_modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True
    if stage == "last":
        blocks = getattr(model, "blocks", getattr(model, "stages", None))
        if blocks is None:
            raise ValueError(f"model {type(model).__name__} has no blocks/stages for fine-stage tuning")
        for block in list(blocks)[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=True):
    training = optimizer is not None
    model.train(training)
    if training:
        # Small batch training: preserve ImageNet BN running statistics.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    true, pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        true.extend(labels.cpu().tolist())
        pred.extend(logits.detach().argmax(1).cpu().tolist())
    metrics = classification_metrics(true, pred, EXPECTED_LABELS)
    return {"loss": total_loss / len(true), "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cassava_training.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", type=Path, help="Override Windows path in dataset_config.yaml (for Kaggle/Colab)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "cassava.csv")
    parser.add_argument("--output-dir", type=Path, help="Run directory (default: checkpoints/experiments/<run-name>)")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pth in the run directory")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--loss", choices=("standard", "weighted"))
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--last-stage-epochs", type=int)
    parser.add_argument("--fine-stage", choices=("last", "all"), default="last")
    parser.add_argument("--fine-lr", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--batch-size", type=int, help="Override configured batch size (reduce on CUDA OOM)")
    parser.add_argument("--pilot-train-per-class", type=int)
    parser.add_argument("--pilot-val-per-class", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config.get("architecture"), str) or not config["architecture"]:
        parser.error("config architecture must be a timm model name")
    resize, crop = int(config["resize"]), int(config["input_size"])
    interpolation_name = config["interpolation"]
    if crop <= 0 or resize < crop or interpolation_name not in ("INTER_AREA", "INTER_CUBIC"):
        parser.error("invalid resize/crop/interpolation")
    interpolation = getattr(cv2, interpolation_name)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch size must be positive")
    if any(x is not None and x < 1 for x in (args.pilot_train_per_class, args.pilot_val_per_class)):
        parser.error("pilot per-class counts must be positive")
    if (args.pilot_train_per_class or args.pilot_val_per_class) and not args.pilot:
        parser.error("pilot count overrides require --pilot")
    dataset_config = yaml.safe_load((ROOT / "configs" / "dataset_config.yaml").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    data_root = args.data_root or Path(dataset_config["data_root"])
    manifest_path = args.manifest
    rows, counts = read_manifest(manifest_path, data_root)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    full_train_counts = np.bincount([int(r["label_index"]) for r in train_rows], minlength=5)
    if args.pilot:
        train_rows = balanced_pilot(train_rows, args.pilot_train_per_class or int(config["pilot_train_per_class"]), seed)
        val_rows = balanced_pilot(val_rows, args.pilot_val_per_class or int(config["pilot_val_per_class"]), seed + 1)
    run_dir = args.output_dir or (ROOT / "checkpoints" / "experiments" / args.run_name)
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"run directory not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and not (run_dir / "last.pth").is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {run_dir / 'last.pth'}")
    batch_size = args.batch_size or int(config["batch_size_cuda"] if device.type == "cuda" else config["batch_size_cpu"])
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    resize_mode = config.get("resize_mode", "square")
    augmentation_profile = config.get("augmentation_profile", "legacy")
    train_crop = config.get("train_crop", "center")
    if resize_mode not in ("square", "shortest_edge"):
        parser.error("resize_mode must be square or shortest_edge")
    if augmentation_profile not in ("legacy", "symptom_preserving", "none"):
        parser.error("unknown augmentation_profile")
    if train_crop not in ("center", "random"):
        parser.error("train_crop must be center or random")
    if config.get("scheduler", "none") not in ("none", "cosine"):
        parser.error("scheduler must be none or cosine")
    min_lr_ratio = float(config.get("scheduler_min_lr_ratio", 0.05))
    if not 0.0 <= min_lr_ratio <= 1.0:
        parser.error("scheduler_min_lr_ratio must be in [0, 1]")
    dataset_args = {"resize": resize, "crop": crop, "interpolation": interpolation,
                    "resize_mode": resize_mode, "augmentation_profile": augmentation_profile,
                    "train_crop": train_crop}
    generator = torch.Generator().manual_seed(seed)
    loader_args = {"batch_size": batch_size, "num_workers": int(config["num_workers"]),
                   "worker_init_fn": seed_worker, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(CassavaDataset(train_rows, data_root, True, seed, **dataset_args), shuffle=True,
                              generator=generator, **loader_args)
    val_loader = DataLoader(CassavaDataset(val_rows, data_root, False, seed, **dataset_args), shuffle=False,
                            **loader_args)
    # Explicit ImageNet initialization; never load the disease-specific checkpoint.
    pretrained_path = ROOT / config["pretrained_file"]
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"ImageNet pretrained weights missing: {pretrained_path}")
    pretrained_hash = hash_file(pretrained_path)
    expected_pretrained_hash = config.get("pretrained_sha256")
    if expected_pretrained_hash and pretrained_hash != expected_pretrained_hash:
        raise ValueError("ImageNet pretrained weights SHA-256 mismatch")
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(str(pretrained_path)), strict=True)
    model.reset_classifier(5)
    model.to(device)
    loss_name = args.loss or config["loss"]
    train_counts = np.bincount([int(r["label_index"]) for r in train_rows], minlength=5)
    # Exponent 1.0 preserves historical inverse-frequency weighting. Lower
    # exponents temper the minority boost to reduce noisy-gradient overfit.
    weight_power = float(config.get("class_weight_power", 1.0))
    if weight_power <= 0:
        raise ValueError("class_weight_power must be positive")
    weights = (sum(full_train_counts) / (5 * full_train_counts)) ** weight_power
    weights = weights.astype(np.float32)
    weight_tensor = torch.tensor(weights, device=device) if loss_name == "weighted" else None
    label_smoothing = (float(config.get("label_smoothing", 0.0))
                       if args.label_smoothing is None else args.label_smoothing)
    if not 0.0 <= label_smoothing < 1.0:
        parser.error("label smoothing must be in [0, 1)")
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    head_epochs = args.head_epochs if args.head_epochs is not None else int(config["head_epochs"])
    last_epochs = args.last_stage_epochs if args.last_stage_epochs is not None else int(config["last_stage_epochs"])
    if args.pilot:
        head_epochs, last_epochs = 1, 0
    run_info = {"run_name": args.run_name, "pilot": args.pilot, "seed": seed,
                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__, "timm": timm.__version__, "opencv": cv2.__version__,
                "numpy": np.__version__, "safetensors": safetensors.__version__,
                "manifest_sha256": hash_file(manifest_path),
                "config": config, "manifest_counts": counts, "train_n": len(train_rows), "val_n": len(val_rows),
                "train_class_counts": train_counts.tolist(), "loss": loss_name,
                "class_weight_power": weight_power if loss_name == "weighted" else None,
                "class_weights": weights.tolist() if loss_name == "weighted" else None,
                "batch_size": batch_size, "amp": use_amp, "schedule": {"head_epochs": head_epochs, "fine_epochs": last_epochs,
                                                        "fine_stage": args.fine_stage,
                                                        "fine_lr": args.fine_lr or config["last_stage_learning_rate"],
                                                        "label_smoothing": label_smoothing,
                                                        "scheduler": config.get("scheduler", "none")},
                "pretrained_file_sha256": pretrained_hash,
                "epochs": [], "best_epoch": None,
                "best_val_macro_f1": -1.0, "pretrained_cfg": model.pretrained_cfg}
    started = time.perf_counter()
    elapsed_offset = 0.0
    epoch = 0
    resume_state = None
    if args.resume:
        resume_state = torch.load(run_dir / "last.pth", map_location="cpu", weights_only=False)
        if resume_state["manifest_sha256"] != hash_file(manifest_path) or resume_state["pretrained_sha256"] != pretrained_hash:
            raise ValueError("Resume manifest or ImageNet weights differ from the original run")
        previous = resume_state["run_info"]
        if (previous["config"] != config or previous["pilot"] != args.pilot or
                previous["batch_size"] != batch_size or previous["schedule"] != run_info["schedule"] or
                previous["train_n"] != len(train_rows) or previous["val_n"] != len(val_rows)):
            raise ValueError("Resume configuration, schedule, batch size or pilot sample changed")
        model.load_state_dict(resume_state["model_state"], strict=True)
        run_info = resume_state["run_info"]
        if run_info["epochs"]:
            elapsed_offset = float(run_info["epochs"][-1]["elapsed_seconds"])
        epoch = int(resume_state["epoch"])
        random.setstate(resume_state["python_rng"])
        np.random.set_state(resume_state["numpy_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        if "loader_rng" in resume_state:
            generator.set_state(resume_state["loader_rng"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
    stages = (("head", head_epochs, config["head_learning_rate"]),
              (args.fine_stage, last_epochs, args.fine_lr or config["last_stage_learning_rate"]))
    start_stage = int(resume_state["stage_index"]) if resume_state else 0
    for stage_index, (stage, n_epochs, lr) in enumerate(stages):
        if stage_index < start_stage:
            continue
        if n_epochs <= 0:
            continue
        if stage != "head" and not (resume_state and stage_index == start_stage):
            best_state = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=True)
            model.load_state_dict(best_state["model_state"], strict=True)
        set_trainable(model, stage)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                       lr=float(lr), weight_decay=float(config["weight_decay"]))
        scheduler_name = config.get("scheduler", "none")
        if scheduler_name not in ("none", "cosine"):
            raise ValueError(f"unsupported scheduler: {scheduler_name}")
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(config.get("scheduler_t_max", n_epochs))),
            eta_min=float(lr) * min_lr_ratio)
                     if scheduler_name == "cosine" else None)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        stale = int(resume_state["stale"]) if resume_state and stage_index == start_stage else 0
        if resume_state and stage_index == start_stage:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            if scheduler is not None and resume_state.get("scheduler_state"):
                scheduler.load_state_dict(resume_state["scheduler_state"])
            if scaler is not None and resume_state.get("scaler_state"):
                scaler.load_state_dict(resume_state["scaler_state"])
        first_epoch = int(resume_state["epoch_in_stage"]) if resume_state and stage_index == start_stage else 0
        for epoch_in_stage in range(first_epoch, n_epochs):
            if stale >= int(config["early_stopping_patience"]):
                break
            epoch += 1
            train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
            val_metrics = run_epoch(model, val_loader, device, criterion, use_amp=use_amp)
            record = {"epoch": epoch, "stage": stage, "train": train_metrics, "val": val_metrics,
                      "elapsed_seconds": round(elapsed_offset + time.perf_counter() - started, 2)}
            run_info["epochs"].append(record)
            print(json.dumps(record), flush=True)
            (run_dir / "epochs.jsonl").open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
            if val_metrics["macro_f1"] > run_info["best_val_macro_f1"] + 1e-6:
                run_info["best_epoch"] = epoch
                run_info["best_val_macro_f1"] = val_metrics["macro_f1"]
                checkpoint = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              "architecture": config["architecture"], "class_names": EXPECTED_LABELS,
                              "preprocessing": {"resize": resize, "crop": crop, "interpolation": interpolation_name,
                                                "resize_mode": resize_mode,
                                                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                              "best_epoch": epoch, "val_macro_f1": val_metrics["macro_f1"]}
                torch.save(checkpoint, run_dir / "best.pth")
                stale = 0
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step()
            (run_dir / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "scaler_state": scaler.state_dict() if scaler is not None else None,
                        "run_info": run_info, "epoch": epoch, "stage_index": stage_index,
                        "epoch_in_stage": epoch_in_stage + 1, "stale": stale,
                        "manifest_sha256": hash_file(manifest_path), "pretrained_sha256": pretrained_hash,
                        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "loader_rng": generator.get_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       run_dir / "last.pth")
            if stale >= int(config["early_stopping_patience"]):
                break
        resume_state = None
    run_info["duration_seconds"] = round(elapsed_offset + time.perf_counter() - started, 2)
    run_info["best_checkpoint_sha256"] = hash_file(run_dir / "best.pth")
    if device.type == "cuda":
        run_info["gpu_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    (run_dir / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": run_info["best_epoch"], "best_val_macro_f1": run_info["best_val_macro_f1"],
                      "best_checkpoint_sha256": run_info["best_checkpoint_sha256"],
                      "duration_seconds": run_info["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
