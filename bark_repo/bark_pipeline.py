# src/tts/bark_pipeline.py

from bark import generate_audio, SAMPLE_RATE
from bark.generation import preload_models
import numpy as np
from scipy.io import wavfile
import librosa
import soundfile as sf
from pathlib import Path
import textwrap
from scipy.signal import butter, filtfilt



# Preload Bark models once
preload_models()



def low_pass_filter(data, cutoff_freq, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def generate_bark_audio(text: str, output_wav_path: str,
                        history_prompt="v2/en_speaker_2",
                        speedup=2, chunk_char_limit=220):
    """
    Generate Bark TTS audio from input text, with chunking, time-stretching, and resampling.
    Saves audio to output_wav_path (.wav at 44.1kHz).
    """

    print(f"Generating audio for: {text[:60]}...")

    # Step 1: Break long text into chunks for Bark stability
    chunks = textwrap.wrap(text, width=chunk_char_limit)

    audio_arrays = []

    for i, chunk in enumerate(chunks):
        print(f" Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        audio = generate_audio(chunk, history_prompt=history_prompt)
        audio_arrays.append(audio)

    # Step 2: Concatenate audio
    combined = np.concatenate(audio_arrays)

    # Step 3: Save at original Bark sample rate (24kHz)
    temp_path = Path(output_wav_path).with_suffix(".temp.wav")
    wavfile.write(temp_path, SAMPLE_RATE, (combined * 32767).astype(np.int16))

    # Step 4: Resample to 44.1kHz + speed up
    y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
    y = low_pass_filter(y, 8000, sr)
    y_stretched = librosa.effects.time_stretch(y, rate=speedup)
    wavfile.write(output_wav_path, 44100, (y_stretched * 32767).astype(np.int16))

    # Step 5: Clean up temp file
    temp_path.unlink(missing_ok=True)

    print(f" Done! Audio saved to {output_wav_path}")
