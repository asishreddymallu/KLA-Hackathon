# KLA HAT Grayscale Super-Resolution Submission

This folder is the minimal submission version derived from the KLA grayscale HAT Colab notebook and matched to the supplied `best.pth` checkpoint.

## Files

```text
KLA_HAT_FINAL_MINIMAL/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── checkpoints/
    └── best.pth
```

The supplied checkpoint is the original full PyTorch checkpoint produced by the notebook. Its learned network parameters are stored under `checkpoint["model"]`.

## Model

- HAT (Hybrid Attention Transformer)
- Grayscale input: 1 channel
- Grayscale output: 1 channel
- Scale: 2×
- embed_dim: 64
- depths: (4, 4, 4, 4)
- num_heads: (4, 4, 4, 4)
- window_size: 8
- patch size: 1
- upsampler: pixelshuffle
- residual connection: 1conv
- L1 loss
- AdamW, learning rate 2e-4
- CosineAnnealingLR
- AMP
- gradient clipping: 1.0
- train/validation split: 90/10
- split seed: 42
- target total epochs: 60

## Google Drive layout

The training scripts expect:

```text
My Drive/
└── KLA_HAT/
    ├── train/
    │   ├── GT/
    │   └── NoisyLR/
    ├── Test_NoisyLR/
    └── checkpoints/
        ├── best.pth
        └── latest.pth   # generated after/resume training
```

The checkpoint included in this submission is:

```text
checkpoints/best.pth
```

## Colab training

Upload `train.py`, `requirements.txt`, and the `checkpoints` folder to Colab. Your Drive dataset remains under `/content/drive/MyDrive/KLA_HAT`.

Run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` is configured for 60 total epochs. It resumes from `checkpoints/latest.pth` when that file exists; otherwise it can resume from `checkpoints/best.pth`. With the supplied checkpoint, the stored epoch is 56, so the next training epoch is 57.

To force a fresh model:

```bash
python train.py --no-resume
```

## Evaluation

Run:

```bash
python evaluate.py
```

By default it loads:

```text
/content/drive/MyDrive/KLA_HAT/checkpoints/best.pth
```

and reads:

```text
/content/drive/MyDrive/KLA_HAT/Test_NoisyLR/*.npy
```

Restored outputs are written to:

```text
/content/drive/MyDrive/KLA_HAT/test_predictions/*.npy
```

To save PNG previews as well:

```bash
python evaluate.py --save-png
```

## Important checkpoint detail

`best.pth` is a full PyTorch training checkpoint, not a plain state dictionary. It contains keys such as:

```text
model
optimizer
scheduler
scaler
epoch
best_psnr
history
```

The actual learned HAT weights are in:

```python
checkpoint["model"]
```

## Supplied checkpoint information

The provided `best.pth` records:

- epoch: 56
- best validation PSNR: 28.08259262918413 dB
- model tensors: 412
- trainable parameter values stored: 1,467,265

No benchmark values beyond those stored in the checkpoint should be inferred from this package.
