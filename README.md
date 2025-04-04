# Voice Deepfake Detection – Take-Home Assessment (Momenta)

## Overview

This project aims to detect voice deepfakes using machine learning. Audio deepfakes are becoming increasingly realistic and pose risks like impersonation and fraud. I implemented a classification model using MFCC features and a simple feedforward neural network to distinguish between real and fake audio samples.

---

## Model Selection

**Model**: Feedforward Neural Network with 3 Fully Connected Layers  
**Reason**: Lightweight, quick to train, and effective for small-scale audio feature classification tasks. Since this is a take-home assignment, the goal was to validate pipeline and logic under resource constraints.

---

## Dataset

- Format: Folder with subfolders `REAL/` and `FAKE/`
- Audio type: `.wav`
- Sampling rate: 16kHz
- Preprocessing:
  - Loaded using `librosa`
  - Extracted 40 MFCC features per audio
  - Averaged across time dimension for fixed-size input vector

---

## Pipeline Steps

1. **Mount Google Drive**: Load dataset from `AUDIO/REAL` and `AUDIO/FAKE`
2. **Preprocess Audio**: Extract MFCCs using `librosa`
3. **Dataset Loader**: `torch.utils.data.Dataset` to handle batch loading
4. **Model Architecture**: 3-layer MLP with ReLU activations and softmax output
5. **Training**:
   - Epochs: 10
   - Optimizer: Adam
   - Loss: CrossEntropy
6. **Prediction**:
   - Single-file inference
   - Confidence scores for both classes

---

## Results

- Final Training Loss: *(Example: 0.15 – will vary)*
- Test Inference Example:
  - Input: Sample fake audio
  - Output: `Predicted Label: FAKE`
  - Confidence Scores: 'REAL=0.7310, FAKE=0.2690'

---

## Challenges Faced

- Dataset imbalance could affect generalization
- Simple MFCC averaging loses temporal resolution
- Limited compute made it harder to use advanced models like CNN or Wav2Vec2

---

## Future Improvements

- Use a CNN or LSTM over MFCC time-steps to retain temporal dynamics
- Fine-tune a pretrained model like Wav2Vec2
- Evaluate on real-world spoofed calls or interviews
- Augment data with noise and pitch-shift for robustness

---

## Conclusion

This project demonstrates the feasibility of detecting voice deepfakes using MFCC-based features and a simple neural network. The architecture can serve as a baseline for future research involving more sophisticated models and datasets.

---

## Run Instructions

```bash
pip install torchaudio librosa numpy matplotlib
