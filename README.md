# KLA HAT Grayscale Super-Resolution

This is the submission-ready split of the original `KLA_HAT_Grayscale_Finetune.ipynb`.

The notebook combined training and test inference in one Colab workflow. Here they are separated into:

- `train.py` — training, validation, checkpoint saving, and resume support.
- `evaluate.py` — inference using a trained checkpoint.

The submission does **not** require Google Drive. All paths are local or supplied through command-line arguments.

## Project structure

```text
KLA_HAT_FINAL_MINIMAL/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── checkpoints/
    └── best.pth
```

`test_predictions/` is created automatically by `evaluate.py`.

For training, the dataset should be arranged as:

```text
project/
├── train/
│   ├── GT/
│   │   ├── image1.npy
│   │   ├── image2.npy
│   │   └── ...
│   └── NoisyLR/
│       ├── image1.npy
│       ├── image2.npy
│       └── ...
└── checkpoints/
```

Each GT/LQ pair must have the same filename stem. The training code expects a 2× relationship, for example:

```text
GT       : 256 × 256
NoisyLR  : 128 × 128
```

The arrays are grayscale `.npy` files.

## Model

The model is the grayscale ×2 HAT configuration used in the original notebook:

- HAT (Hybrid Attention Transformer)
- 1 input channel
- 1 output channel
- 2× super-resolution
- `embed_dim=64`
- `depths=(4, 4, 4, 4)`
- `num_heads=(4, 4, 4, 4)`
- `window_size=8`
- `patch_size=1`
- PixelShuffle upsampler
- `1conv` residual connection
- L1 loss
- AdamW
- Initial learning rate: `2e-4`
- Cosine annealing learning-rate schedule
- AMP
- Gradient clipping: `1.0`
- 90/10 train/validation split
- Split seed: `42`
- Training crop: `128 × 128` GT / `64 × 64` LR
- Default total epochs: `100`
- Default batch size: `4`

## Training

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then place the training dataset in:

```text
train/GT/
train/NoisyLR/
```

Run:

```bash
python train.py
```

The script uses a local project root by default.

To specify a different project/data root:

```bash
python train.py --project /path/to/project
```

The script creates:

```text
checkpoints/latest.pth
checkpoints/best.pth
```

When `latest.pth` exists, training resumes from it. If `latest.pth` is not present but `best.pth` exists, the script can resume from `best.pth`.

To force a fresh run:

```bash
python train.py --no-resume
```

The training script requires an NVIDIA GPU with CUDA.

## Inference / evaluation

The included `checkpoints/best.pth` is the trained checkpoint from the original notebook.

The checkpoint is a full PyTorch training checkpoint. The learned HAT parameters are stored under:

```python
checkpoint["model"]
```

Place test `.npy` files in:

```text
Test_NoisyLR/
```

Then run:

```bash
python evaluate.py
```

The restored 2× grayscale outputs are written to:

```text
test_predictions/
```

Each output is saved as `.npy` with the same filename as its input.

To also generate 8-bit grayscale PNG previews:

```bash
python evaluate.py --save-png
```

You can also supply paths explicitly:

```bash
python evaluate.py \
    --checkpoint checkpoints/best.pth \
    --test-dir /path/to/Test_NoisyLR \
    --output-dir /path/to/test_predictions
```

## Supplied checkpoint

The included checkpoint records:

```text
epoch: 56
best validation PSNR: 28.08259262918413 dB
model tensors: 412
trainable parameter values: 1,467,265
```

The checkpoint's `epoch=56` is the zero-based epoch index stored by the training loop.

## Notes

`train.py` and `evaluate.py` automatically clone the official HAT repository when it is not already available locally, then install the HAT/BasicSR dependencies.

The scripts also apply the BasicSR `rgb_to_grayscale` import compatibility patch used in the original notebook.

The original notebook is not required to run inference. The included `best.pth` is sufficient for evaluation.
