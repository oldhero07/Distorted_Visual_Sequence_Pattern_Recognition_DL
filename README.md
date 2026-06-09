# Distorted Visual Sequence Pattern Recognition



The task is to read 6-character text sequences from heavily distorted grayscale images — characters overlap, blur, and have uneven alignment. Standard OCR fails here because there's no clean segmentation. I solved it using a CRNN trained with CTC loss, which handles variable-length alignment without needing character-level annotations.

## Approach

I went with a ResNet-CRNN architecture. The idea is straightforward: a CNN extracts visual features from the image, those features get reshaped into a time sequence, and a BiLSTM reads that sequence to predict characters left-to-right and right-to-left simultaneously. CTC loss then figures out the alignment during training.

The reason I chose residual blocks over a plain CNN is that skip connections let gradients flow cleanly through deeper layers. Without them, the model struggles to learn fine character details under heavy noise.

## Architecture

```
Input (1 × 32 × 200 grayscale)
    ↓
ResNet CNN Backbone
  - init conv: 1 → 32 channels
  - 4 residual stages: 32→64→128→256→512
  - MaxPool after each stage (height halved, width preserved later)
  - AdaptiveAvgPool2d((1, None)) → squash height to 1
    ↓
Sequence: 50 time steps × 512 features
    ↓
2-layer Bidirectional LSTM (hidden=256, dropout=0.3)
    ↓
FC layer → 39 classes (38 chars + CTC blank)
    ↓
CTC Loss / Beam Search Decoding
```

## Training

| Setting | Value |
|---|---|
| Image size | 32 × 200 |
| Batch size | 64 |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler | OneCycleLR — 10% warmup + cosine decay |
| Gradient clipping | 5.0 |
| Epochs | 50 |
| Train / Val split | 18,000 / 2,000 |

**Augmentations:** RandomAffine, RandomPerspective, GaussianBlur, ColorJitter, RandomErasing — chosen to mimic the kinds of distortions actually present in the images.

I used OneCycleLR instead of a fixed LR or ReduceLROnPlateau because it warms up first (avoids bad early updates) then decays smoothly. In practice it converged noticeably faster.

## Results

| Metric | Value |
|---|---|
| Val CER | 0.0007 |
| Exact match accuracy | 99.80% (1996 / 2000) |

Both greedy and beam search (width=10) decoding gave identical results — meaning the model's output distributions are already very sharp, so there's no ambiguity left for beam search to resolve.

## Files

- `notebook_Himesh_Kumar_24119021.ipynb` — full pipeline: EDA, model, training, evaluation, test predictions
- `submission_Himesh_Kumar_24119021.csv` — predictions for all 5,000 test images

## Running in Colab

1. Upload `cig_ps.zip` to your Google Drive
2. Open the notebook in Google Colab with a T4 GPU runtime
3. Run all cells — Drive mounts automatically, zip is found and extracted, training starts
4. The submission CSV downloads automatically after training
