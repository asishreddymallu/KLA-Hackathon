#!/usr/bin/env python3
"""
KLA HAT grayscale x2 super-resolution training script.

This script is the standalone training part extracted from the
KLA_HAT_Grayscale_Finetune notebook. It does not use Google Drive.

Expected dataset layout:

project/
├── train/
│   ├── GT/
│   │   ├── image1.npy
│   │   └── ...
│   └── NoisyLR/
│       ├── image1.npy
│       └── ...
└── checkpoints/

Run:
    python train.py

To use another project/data location:
    python train.py --project /path/to/project

The script resumes from checkpoints/latest.pth when present.
If there is no latest checkpoint, it resumes from checkpoints/best.pth.
Use --no-resume to start a fresh training run.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def install_hat_if_needed() -> None:
    """Clone the official HAT repository and install its dependencies."""
    hat_dir = Path("./HAT").resolve()

    if not hat_dir.exists():
        print(f"Cloning HAT into {hat_dir}")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/XPixelGroup/HAT.git",
                str(hat_dir),
            ],
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "basicsr==1.3.4.9",
            "einops",
            "lmdb",
            "addict",
            "yapf",
        ],
        check=True,
    )

    subprocess.run(
        [sys.executable, "setup.py", "develop", "-q"],
        cwd=hat_dir,
        check=True,
    )


def patch_basicsr() -> None:
    """Apply the torchvision compatibility patch used by the notebook."""
    candidates = [
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "basicsr"
        / "data"
        / "degradations.py",
    ]

    for path in candidates:
        if not path.exists():
            continue

        text = path.read_text()
        old = (
            "from torchvision.transforms.functional_tensor "
            "import rgb_to_grayscale"
        )
        new = (
            "from torchvision.transforms.functional "
            "import rgb_to_grayscale"
        )

        if old in text:
            path.write_text(text.replace(old, new))
            print(f"Patched BasicSR: {path}")
        return


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the grayscale x2 KLA HAT model."
    )
    p.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Project/data root containing train/ and checkpoints/.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total number of epochs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start from scratch.",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip HAT/BasicSR installation.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(42)

    if not args.skip_install:
        install_hat_if_needed()
        patch_basicsr()

    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from hat.archs.hat_arch import HAT

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Use an NVIDIA GPU runtime."
        )

    device = torch.device("cuda")
    project = args.project.resolve()

    gt_dir = project / "train" / "GT"
    lq_dir = project / "train" / "NoisyLR"
    ckpt_dir = project / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scale = 2
    gt_patch_size = 128
    val_ratio = 0.10
    split_seed = 42
    weight_decay = 0.0
    grad_clip = 1.0
    use_amp = True

    if not gt_dir.exists() or not lq_dir.exists():
        raise FileNotFoundError(
            "Expected dataset folders:\n"
            f"  {gt_dir}\n"
            f"  {lq_dir}"
        )

    gt_files = {p.stem: p for p in gt_dir.glob("*.npy")}
    lq_files = {p.stem: p for p in lq_dir.glob("*.npy")}
    common = sorted(set(gt_files) & set(lq_files))

    if not common:
        raise RuntimeError("No paired .npy files found.")

    first_gt = np.load(gt_files[common[0]])
    first_lq = np.load(lq_files[common[0]])

    if first_gt.ndim != 2 or first_lq.ndim != 2:
        raise ValueError(
            "Dataset arrays must be 2-D grayscale arrays."
        )

    expected_gt_shape = (
        first_lq.shape[0] * scale,
        first_lq.shape[1] * scale,
    )
    if first_gt.shape != expected_gt_shape:
        raise ValueError(
            f"Expected 2x pairs, got GT={first_gt.shape}, "
            f"LQ={first_lq.shape}"
        )

    print(f"Paired samples: {len(common)}")
    print(f"GT example:  {first_gt.shape}")
    print(f"LQ example:  {first_lq.shape}")

    class KLAGrayPairedDataset(Dataset):
        def __init__(self, pairs, phase="train"):
            self.pairs = pairs
            self.phase = phase

        @staticmethod
        def _load(path: Path) -> np.ndarray:
            x = np.load(path).astype(np.float32)

            if x.ndim == 3:
                if x.shape[-1] == 1:
                    x = x[..., 0]
                elif x.shape[0] == 1:
                    x = x[0]
                else:
                    raise ValueError(
                        f"Expected grayscale array, got {x.shape}: {path}"
                    )

            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2-D grayscale array, got {x.shape}: {path}"
                )

            return x

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            gt_path, lq_path = self.pairs[idx]

            gt = self._load(gt_path)
            lq = self._load(lq_path)

            if gt.shape != (
                lq.shape[0] * scale,
                lq.shape[1] * scale,
            ):
                raise ValueError(
                    f"Shape mismatch: GT={gt.shape}, LQ={lq.shape}"
                )

            if self.phase == "train":
                ps = gt_patch_size
                lp = ps // scale

                if gt.shape[0] < ps or gt.shape[1] < ps:
                    raise ValueError(
                        "GT image is smaller than the requested "
                        "training crop."
                    )

                top = random.randint(0, gt.shape[0] - ps)
                left = random.randint(0, gt.shape[1] - ps)

                gt = gt[top:top + ps, left:left + ps]
                lq = lq[
                    top // scale: top // scale + lp,
                    left // scale: left // scale + lp,
                ]

                if random.random() < 0.5:
                    gt = np.flip(gt, axis=1).copy()
                    lq = np.flip(lq, axis=1).copy()

                if random.random() < 0.5:
                    gt = np.flip(gt, axis=0).copy()
                    lq = np.flip(lq, axis=0).copy()

                if random.random() < 0.5:
                    gt = np.rot90(gt).copy()
                    lq = np.rot90(lq).copy()

            return {
                "lq": torch.from_numpy(lq[None, ...].copy()).float(),
                "gt": torch.from_numpy(gt[None, ...].copy()).float(),
                "name": gt_path.stem,
            }

    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(common))
    rng.shuffle(indices)

    val_count = max(1, int(len(indices) * val_ratio))
    val_ids = indices[:val_count]
    train_ids = indices[val_count:]

    pairs = [(gt_files[n], lq_files[n]) for n in common]
    train_pairs = [pairs[i] for i in train_ids]
    val_pairs = [pairs[i] for i in val_ids]

    train_ds = KLAGrayPairedDataset(train_pairs, "train")
    val_ds = KLAGrayPairedDataset(val_pairs, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = HAT(
        img_size=gt_patch_size // scale,
        patch_size=1,
        in_chans=1,
        embed_dim=64,
        depths=(4, 4, 4, 4),
        num_heads=(4, 4, 4, 4),
        window_size=8,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=scale,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    scaler = GradScaler(enabled=use_amp)

    latest = ckpt_dir / "latest.pth"
    best = ckpt_dir / "best.pth"

    start_epoch = 0
    best_psnr = -float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "lr": [],
    }

    if not args.no_resume:
        resume_path = latest if latest.exists() else (
            best if best.exists() else None
        )

        if resume_path is not None:
            state = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )

            model.load_state_dict(state["model"])

            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])

            start_epoch = int(state.get("epoch", -1)) + 1
            best_psnr = float(
                state.get("best_psnr", best_psnr)
            )
            history = state.get("history", history)

            print(
                f"Resumed from {resume_path} "
                f"at epoch {start_epoch + 1}/{args.epochs}; "
                f"best PSNR={best_psnr:.3f} dB"
            )

    def validate():
        model.eval()
        total_loss = 0.0
        psnrs = []

        with torch.no_grad():
            for batch in val_loader:
                lq = batch["lq"].to(
                    device,
                    non_blocking=True,
                )
                gt = batch["gt"].to(
                    device,
                    non_blocking=True,
                )

                with autocast(
                    enabled=use_amp,
                    dtype=torch.float16,
                ):
                    pred = model(lq)
                    loss = criterion(pred, gt)

                pred = pred.float().clamp(0, 1)
                gt = gt.float().clamp(0, 1)

                mse = torch.mean((pred - gt) ** 2).item()
                psnr = (
                    99.0
                    if mse <= 1e-12
                    else -10.0 * np.log10(mse)
                )

                total_loss += loss.item()
                psnrs.append(psnr)

        return (
            total_loss / max(1, len(val_loader)),
            float(np.mean(psnrs)),
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for batch in pbar:
            lq = batch["lq"].to(
                device,
                non_blocking=True,
            )
            gt = batch["gt"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                enabled=use_amp,
                dtype=torch.float16,
            ):
                pred = model(lq)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = running / max(1, len(train_loader))
        val_loss, val_psnr = validate()

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["lr"].append(
            optimizer.param_groups[0]["lr"]
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": max(best_psnr, val_psnr),
            "history": history,
        }

        torch.save(state, latest)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state["best_psnr"] = best_psnr
            torch.save(state, best)

        print(
            f"epoch={epoch + 1} "
            f"train_l1={train_loss:.6f} "
            f"val_l1={val_loss:.6f} "
            f"val_PSNR={val_psnr:.3f} dB"
        )

    print(f"Training finished.")
    print(f"Latest checkpoint: {latest}")
    print(f"Best checkpoint:    {best}")


if __name__ == "__main__":
    main()
