import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from scipy.signal import butter, lfilter

# --- Helper functions ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def detect_pitch_autocorr(y, sr, frame_length=2048, hop_length=512):
    pitches = []
    times = []
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i:i+frame_length]
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        d = np.diff(corr)
        pos_slope = np.where(d > 0)[0]
        if len(pos_slope) == 0:
            pitches.append(0)  # No detectable pitch
            times.append(i / sr)
            continue

        start = pos_slope[0]
        peak = np.argmax(corr[start:]) + start
        pitch = sr / peak if peak != 0 else 0
        pitches.append(pitch if pitch < 500 else 0)
        times.append(i / sr)

    return np.array(times), np.array(pitches)


# --- Streamlit App ---
st.set_page_config(page_title="Voice Pitch Detection and Visualization")
st.title("Voice Pitch Detection and Visualization")

uploaded_file = st.file_uploader("Upload a recorded audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)

    # Preprocessing
    y = bandpass_filter(y, lowcut=80, highcut=300, fs=sr)
    y = librosa.effects.preemphasis(y)

    # Pitch detection
    times, pitches = detect_pitch_autocorr(y, sr)

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title='Waveform')
    ax[0].label_outer()

    ax[1].plot(times, pitches, label='Pitch (Hz)', color='r')
    ax[1].set(title='Estimated Pitch Over Time', xlabel='Time (s)', ylabel='Pitch (Hz)')
    ax[1].legend()

    st.pyplot(fig)
    st.success("Pitch detection complete.")
