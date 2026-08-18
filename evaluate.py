#!/usr/bin/env python3
"""
KLA HAT grayscale x2 inference/evaluation script.

This is the standalone inference part extracted from the
KLA_HAT_Grayscale_Finetune notebook. It does not use Google Drive.

Default project layout:

project/
├── checkpoints/
│   └── best.pth
├── Test_NoisyLR/
│   ├── image1.npy
│   └── ...
└── test_predictions/

Run:
    python evaluate.py

Optional PNG previews:
    python evaluate.py --save-png

Custom paths:
    python evaluate.py \
        --checkpoint /path/to/best.pth \
        --test-dir /path/to/Test_NoisyLR \
        --output-dir /path/to/test_predictions
"""

from __future__ import annotations

import argparse
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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run grayscale x2 HAT inference on .npy files."
    )
    p.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Project root containing checkpoints/ and test data.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to PROJECT/checkpoints/best.pth.",
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Directory containing test .npy files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for restored .npy files.",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip HAT/BasicSR installation.",
    )
    p.add_argument(
        "--save-png",
        action="store_true",
        help="Also save normalized grayscale PNG previews.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if not args.skip_install:
        install_hat_if_needed()
        patch_basicsr()

    import torch
    from torch.cuda.amp import autocast
    from tqdm.auto import tqdm
    from hat.archs.hat_arch import HAT

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Use an NVIDIA GPU runtime."
        )

    device = torch.device("cuda")
    project = args.project.resolve()

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else project / "checkpoints" / "best.pth"
    )

    test_dir = (
        args.test_dir
        if args.test_dir is not None
        else project / "Test_NoisyLR"
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else project / "test_predictions"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir.resolve()}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "Expected a full checkpoint containing checkpoint['model']."
        )

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Checkpoint best PSNR: {checkpoint.get('best_psnr')}")

    model = HAT(
        img_size=64,
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
        upscale=2,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    model.eval()

    files = sorted(test_dir.glob("*.npy"))

    if not files:
        raise RuntimeError(
            f"No .npy test files found in {test_dir.resolve()}"
        )

    with torch.no_grad():
        for path in tqdm(files, desc="Evaluating"):
            arr = np.load(path).astype(np.float32)

            if arr.ndim == 3 and 1 in arr.shape:
                if arr.shape[-1] == 1:
                    arr = arr[..., 0]
                elif arr.shape[0] == 1:
                    arr = arr[0]

            if arr.ndim != 2:
                raise ValueError(
                    f"Expected grayscale HxW input, "
                    f"got {arr.shape}: {path}"
                )

            x = torch.from_numpy(
                arr[None, None]
            ).to(device)

            with autocast(
                enabled=True,
                dtype=torch.float16,
            ):
                pred = model(x)

            pred = (
                pred.float()
                .clamp(0, 1)[0, 0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            np.save(
                output_dir / path.name,
                pred,
            )

            if args.save_png:
                from PIL import Image

                out = (
                    pred * 255.0
                ).round().clip(
                    0, 255
                ).astype(np.uint8)

                Image.fromarray(
                    out,
                    mode="L",
                ).save(
                    output_dir / f"{path.stem}.png"
                )

    print()
    print(f"Saved {len(files)} restored arrays to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
