#Step 1: Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')

#Step 2: Define Dataset Paths
#dataset is located in the AUDIO folder, with subfolders REAL and FAKE
import os

dataset_path = "/content/drive/MyDrive/AUDIO"  # Adjust the path if necessary
real_path = os.path.join(dataset_path, "REAL")
fake_path = os.path.join(dataset_path, "FAKE")

print(f"Real samples: {len(os.listdir(real_path))}")
print(f"Fake samples: {len(os.listdir(fake_path))}")

#Step 3: Install Required Libraries
#torchaudio, librosa, numpy, matplotlib

!pip install torchaudio librosa numpy matplotlib

#Step 4: Preprocess the Audio Data
#Convert audio files into spectrograms or Mel-frequency cepstral coefficients (MFCCs).

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)  # Taking mean across time axis

# Example
sample_file = os.path.join(real_path, os.listdir(real_path)[0])
features = extract_features(sample_file)

plt.plot(features)
plt.title("MFCC Features of Sample Audio")
plt.show()

#Step 5: Create Dataset for Training

import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, data_path, label):
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        features = extract_features(file_path)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(self.label, dtype=torch.long)

# Load datasets
real_dataset = AudioDataset(real_path, label=0)  # Real label: 0
fake_dataset = AudioDataset(fake_path, label=1)  # Fake label: 1

# Combine datasets
full_dataset = real_dataset + fake_dataset
train_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)

# Test data loading
for x, y in train_loader:
    print("Batch Features:", x.shape)
    print("Batch Labels:", y)
    break

#Step 6: Define a Deep Learning Model(neural network)

import torch.nn as nn
import torch.optim as optim

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output: 2 classes (Real, Fake)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Initialize model
model = AudioClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Step 7: Train the Model

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

#Step 8: Save the Model

torch.save(model.state_dict(), "/content/drive/MyDrive/audio_classifier.pth")
print("Model saved successfully!")

#Step 9: Load the Saved Model

#reload the trained model from Google Drive

model_path = "/content/drive/MyDrive/audio_classifier.pth"

# Initialize model
model = AudioClassifier()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

print("Model loaded successfully!")

#Step 10: Prepare a Test Audio File
#Choose an audio file from the dataset or upload a new one.
#Use an Existing File

test_file = os.path.join(fake_path, os.listdir(fake_path)[0])  # Test with a FAKE sample
print(f"Testing with: {test_file}")


#Step 11: Extract Features from the Test File
#Use the same feature extraction function as before.

test_features = extract_features(test_file)
test_tensor = torch.tensor(test_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

#Step 12: Make a Prediction
#Pass the extracted features through the model to get predictions.

with torch.no_grad():
    output = model(test_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

# Interpret the result
label_map = {0: "REAL", 1: "FAKE"}
print(f"Predicted Label: {label_map[predicted_label]}")

#Step 13: Display Model Confidence
#You can also view the confidence scores.

probabilities = torch.softmax(output, dim=1).numpy()[0]
print(f"Confidence Scores: REAL={probabilities[0]:.4f}, FAKE={probabilities[1]:.4f}")
