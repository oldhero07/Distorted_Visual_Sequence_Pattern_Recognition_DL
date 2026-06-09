# Distorted Visual Sequence Recognition

This repository contains a deep learning pipeline built in PyTorch to recognize text sequences from distorted grayscale images. The images contain overlapping characters, background noise, blur, and occlusion.

## Model Architecture

The solution uses a **ResNet-CRNN** (Convolutional Recurrent Neural Network) architecture trained with CTC Loss:

1. **ResNet CNN Backbone**: A custom 4-stage residual network with skip connections to prevent vanishing gradients. The channel progression is `32 -> 64 -> 128 -> 256 -> 512`. Each stage uses 3x3 convolutions with batch normalization and ReLU activations.
2. **Height Pooling**: An adaptive average pooling layer (`nn.AdaptiveAvgPool2d((1, None))`) collapses the vertical height of the feature map to 1, converting the 2D spatial features into a 1D sequence of 50 time-steps.
3. **Bidirectional LSTM**: A 2-layer BiLSTM with hidden size 256 and 30% dropout models the sequential character context in both directions.
4. **CTC Loss and Decoding**: Trained using PyTorch's `nn.CTCLoss`. At inference time, I use both Greedy and Beam Search (beam width 10) decoding strategies.

## Training Strategy

- **Image size**: 32x200 grayscale.
- **Augmentation**: Random affine transforms, random perspective distortion, Gaussian blur, color jitter, and random erasing to simulate real-world distortions.
- **Optimizer**: AdamW with weight decay of 1e-4.
- **Scheduler**: OneCycleLR with a max learning rate of 3e-4 and 10% warmup steps followed by cosine decay. This provides significantly faster convergence than a fixed learning rate.
- **Gradient Clipping**: Gradients are clipped to a norm of 5.0 to prevent explosion.

## Evaluation

Performance is measured using Character Error Rate (CER), based on the Levenshtein (edit) distance between predicted and ground truth sequences. Both greedy and beam search decoding produce identical results in my runs, suggesting the model output distributions are already sharply peaked.

## Repository Structure

- `notebook_Himesh_Kumar_24119021.ipynb`: Full pipeline including EDA, model definition, training loop, evaluation, and test inference.
- `submission_Himesh_Kumar_24119021.csv`: Final test set predictions.

## Setup

1. Install dependencies:
   ```bash
   pip install torch torchvision pandas matplotlib scikit-learn pillow tqdm
   ```

2. Run the notebook in Google Colab with a T4 GPU runtime. Upload `cig_ps.zip` to your Google Drive — the notebook will automatically mount Drive, locate the zip, and extract it.
