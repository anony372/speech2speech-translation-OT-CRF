# Install the missing package
!pip install sklearn-crfsuite
!pip install pot

# Import necessary modules
import numpy as np
from transformers import AutoProcessor, SeamlessM4Tv2Model
from sklearn_crfsuite import CRF  # For CRF sequence modeling
from ot import emd2  # Optimal Transport distance function
from scipy.spatial.distance import cdist
import os
!pip install --upgrade numpy tensorflow transformers
# !pip uninstall numpy tensorflow transformers
# !pip install numpy tensorflow transformers
!pip install transformers torchaudio sklearn scipy huggingface_hub

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model, AutoProcessor, SeamlessM4Tv2Model
from sklearn_crfsuite import CRF
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from IPython.display import Audio
import os
from scipy.interpolate import interp1d
from huggingface_hub import login

login(token=~"__")
# Step 1: Install the necessary packages and dependencies
!pip install torchaudio soundfile transformers

# Import required modules
import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel, Wav2Vec2Model
import soundfile as sf
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from IPython.display import Audio, display  # For playing audio in notebook

# Initialize processor and model for translation
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
translation_model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

# Initialize Wav2Vec2 model for encoding
transformer_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')

# Load Tacotron2 model for acoustic decoding
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

# Function to translate input audio using SeamlessM4T model
def translate_audio(encoded_features, target_language_code):
    # Generate translated speech in the target language
    audio_array_from_audio = translation_model.generate(inputs=encoded_features, tgt_lang=target_language_code)[0].cpu().numpy().squeeze()
    return audio_array_from_audio

# Function to save and display audio
def save_and_display_audio(audio_array, filename, sample_rate=16000, title="Audio"):
    # Save audio to a file
    sf.write(filename, audio_array, sample_rate)
    # Display the audio player in the notebook
    print(f"{title}:")
    display(Audio(filename))

# Function to extract features
def extract_features(raw_speech):
    if len(raw_speech.shape) == 1:
        raw_speech = raw_speech.unsqueeze(0)
    return raw_speech

# Function to encode features
def encode_features(raw_waveform, model):
    if len(raw_waveform.shape) == 1:
        raw_waveform = raw_waveform.unsqueeze(0)
    encoded_features = model(raw_waveform).last_hidden_state
    return encoded_features

# Function for CRF sequence modeling
def crf_sequence_modeling(encoded_features):
    crf_model = CRF(algorithm='lbfgs')
    encoded_features_np = encoded_features.detach().cpu().numpy()
    num_time_steps = encoded_features_np.shape[1]
    num_features = encoded_features_np.shape[2]
    labels = ['O'] * num_time_steps
    encoded_features_np = encoded_features_np.reshape(num_time_steps, num_features)
    crf_model.fit([encoded_features_np], [labels])
    predicted_sequence = crf_model.predict([encoded_features_np])[0]
    return predicted_sequence

# Function for acoustic decoding
def acoustic_decoder(predicted_sequence, tacotron2):
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    sequences, lengths = utils.prepare_input_sequence(predicted_sequence)
    with torch.no_grad():
        mel_spectrogram, _, _ = tacotron2.infer(sequences, lengths)
    mel_spectrogram = mel_spectrogram.squeeze(-1)
    return mel_spectrogram

def ensure_2d(array):
    if isinstance(array, torch.Tensor):
        array = array.numpy()
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array

def match_feature_dim(mel_spectrogram_np, reference_distribution_np):
    mel_features = mel_spectrogram_np.shape[1]
    ref_features = reference_distribution_np.shape[1]

    if mel_features < ref_features:
        padding = ref_features - mel_features
        mel_spectrogram_np = np.pad(mel_spectrogram_np, ((0, 0), (0, padding)), mode='constant')
    elif mel_features > ref_features:
        mel_spectrogram_np = mel_spectrogram_np[:, :ref_features]

    return mel_spectrogram_np

# Function for optimal transport alignment
def optimal_transport_alignment(mel_spectrogram, reference_distribution):
    mel_spectrogram_np = mel_spectrogram.cpu().numpy()
    reference_distribution_np = reference_distribution

    mel_spectrogram_np = ensure_2d(mel_spectrogram_np)
    reference_distribution_np = ensure_2d(reference_distribution_np)

    mel_spectrogram_np = match_feature_dim(mel_spectrogram_np, reference_distribution_np)

    cost_matrix = cdist(mel_spectrogram_np, reference_distribution_np, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    aligned_spectrogram = np.zeros_like(reference_distribution_np)
    for i, j in zip(row_ind, col_ind):
        aligned_spectrogram[j, :] = mel_spectrogram_np[i, :]

    return torch.tensor(aligned_spectrogram).float()

# Step 7: Combine and execute the process

# Define the source and target languages
source_language_code = "spa"  # Source language code
target_language_code = "eng"  # Target language code

# Load the input audio
input_audio_path = "/content/cvss_de1.wav"
audio, orig_freq = torchaudio.load(input_audio_path)
audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

# Feature extraction
audio_features = extract_features(audio)

# Encode features
encoded_features = encode_features(audio_features, transformer_model)

# Translate the audio
translated_audio = translate_audio(encoded_features, target_language_code)

# Save and display the original and translated audio
save_and_display_audio(audio.squeeze().numpy(), 'source_audio.wav', title="Source Audio")
save_and_display_audio(translated_audio, 'translated_audio.wav', title="Translated Audio")

# Convert translated audio to tensor for further processing
translated_audio_tensor = torch.tensor(translated_audio).unsqueeze(0)

# Feature extraction for translated audio
translated_audio_features = extract_features(translated_audio_tensor)
encoded_translated_features = encode_features(translated_audio_features, transformer_model)

# CRF sequence modeling
predicted_sequence = crf_sequence_modeling(encoded_translated_features)

# Acoustic decoding using Tacotron2
mel_spectrogram = acoustic_decoder(predicted_sequence, tacotron2)

# Example reference distribution
reference_distribution = np.random.rand(260, 80)

# Optimal transport alignment
aligned_mel_spectrogram = optimal_transport_alignment(mel_spectrogram, reference_distribution)

# Generate synthetic audio from aligned mel-spectrogram using a vocoder
# Assuming a vocoder model is available
vocoder_model = Vocoder.load_model('waveglow_model.pth')  # Placeholder for actual vocoder model
synthetic_audio = vocoder_synthesize(aligned_mel_spectrogram, vocoder_model)

# Save and display the synthetic audio
save_and_display_audio(synthetic_audio.numpy(), 'synthetic_audio.wav', title="Synthetic Audio")

# Plot the aligned mel spectrogram
plt.figure(figsize=(10, 12))
plt.imshow(aligned_mel_spectrogram.detach().cpu().numpy(), origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Aligned Mel Spectrogram in Target Language')
plt.colorbar(label='Amplitude')
plt.show()
