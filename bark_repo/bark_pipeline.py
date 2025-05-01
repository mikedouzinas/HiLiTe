from bark import generate_audio, SAMPLE_RATE
from bark.generation import preload_models
import numpy as np
from scipy.io import wavfile
import librosa
import textwrap
from pathlib import Path
from scipy.signal import butter, filtfilt

# Preload Bark model once at import
preload_models()


def low_pass_filter(data, cutoff_freq, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def generate_bark_audio(text: str, output_wav_path: str,
                        history_prompt="v2/en_speaker_2",
                        speedup=1.25,
                        chunk_char_limit=350):
    """
    Generate Bark TTS audio from input text with:
    - Text chunking for stability
    - Low-pass denoising
    - Time-stretching (with pitch preservation)
    - Resample to 44.1kHz
    """
    print(f"[Bark] Generating audio for: {text[:60]}...")

    # Step 1: Chunk the text
    chunks = textwrap.wrap(text, width=chunk_char_limit)
    audio_arrays = []

    for i, chunk in enumerate(chunks):
        print(f" [Chunk {i+1}/{len(chunks)}] {chunk[:50]}...")
        audio = generate_audio(chunk, history_prompt=history_prompt)
        audio_arrays.append(audio)

    # Step 2: Concatenate generated audio (Bark outputs at 24kHz)
    combined = np.concatenate(audio_arrays)

    # Step 3: Save raw output to temp WAV (24kHz)
    temp_path = Path(output_wav_path).with_suffix(".temp.wav")
    wavfile.write(temp_path, SAMPLE_RATE, (combined * 32767).astype(np.int16))

    # Step 4: Resample to 44.1kHz
    y_24k, _ = librosa.load(temp_path, sr=SAMPLE_RATE)
    y_resampled = librosa.resample(y_24k, orig_sr=SAMPLE_RATE, target_sr=44100)

    # Step 5: Apply low-pass filter
    y_denoised = low_pass_filter(y_resampled, cutoff_freq=8000, fs=44100)

    # Step 6: Stretch to match 14 seconds exactly
    def stretch_audio_to_duration(audio_array, orig_sr, target_duration):
        current_duration = len(audio_array) / orig_sr
        stretch_rate = current_duration / target_duration
        return librosa.effects.time_stretch(audio_array, rate=stretch_rate)

    y_stretched = stretch_audio_to_duration(y_denoised, orig_sr=44100, target_duration=14.0)

    # Step 7: Save final audio
    wavfile.write(output_wav_path, 44100, (y_stretched * 32767).astype(np.int16))

    # Clean up
    temp_path.unlink(missing_ok=True)
    print(f"[Bark] Done! Saved audio to {output_wav_path}")
