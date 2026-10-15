import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file
y, sr = librosa.load("sample_audio.wav", sr=None)

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Display MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()