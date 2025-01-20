import os
import subprocess
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup, low_pass_filter
import librosa
import soundfile as sf
import streamlit as st
import imageio_ffmpeg as ffmpeg_lib
import yt_dlp

# Get FFmpeg path from imageio_ffmpeg
ffmpeg_path = ffmpeg_lib.get_ffmpeg_exe()

# Explicitly set paths for ffmpeg and ffprobe
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffmpeg_path  # Ensure ffprobe points to the correct binary

# Add ffmpeg and ffprobe paths to environment variables
os.environ["FFMPEG"] = ffmpeg_path
os.environ["FFPROBE"] = ffmpeg_path

st.title("Video to Audio Processor with Reverb and Pitch Shifting")

# Function to download audio using yt-dlp
def download_audio(video_url):
    output_file = "video_audio.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_file,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "ffmpeg_location": ffmpeg_path,  # Set FFmpeg path for yt-dlp
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        "verbose": True,  # Enable verbose logging for debugging
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        # Handle double `.mp3` extension
        if os.path.exists(f"{output_file}.mp3"):
            os.rename(f"{output_file}.mp3", output_file)
        return output_file
    except Exception as e:
        raise RuntimeError(f"yt-dlp failed: {e}")

# Input for video URL
video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    st.write("Processing the video...")
    try:
        # Step 1: Download video audio using yt-dlp
        audio_file = download_audio(video_url)

        # Step 2: Load the audio with pydub
        # Explicitly set ffprobe and ffmpeg paths during loading
        audio = AudioSegment.from_file(audio_file, ffmpeg=ffmpeg_path)

        # Step 3: Slow down to 0.8x speed
        audio = speedup(audio, playback_speed=0.8)

        # Step 4: Apply a low-pass filter
        audio = low_pass_filter(audio, cutoff=3000)

        # Export intermediate audio for further processing
        intermediate_file = "intermediate_audio.wav"
        audio.export(intermediate_file, format="wav")

        # Step 5: Add reverb and pitch shift using librosa
        y, sr = librosa.load(intermediate_file, sr=None)

        # Apply pitch shifting (down by 1 semitone)
        y_pitch_shifted = librosa.effects.pitch_shift(y, sr, n_steps=-1)

        # Apply reverb (simple exponential decay emulation)
        reverb_decay = 0.3
        reverb = np.convolve(y_pitch_shifted, np.ones(int(sr * reverb_decay)) / sr)
        y_processed = np.concatenate((y_pitch_shifted, reverb[:len(y_pitch_shifted)]))

        # Save the final processed audio
        final_audio_file = "processed_audio.wav"
        sf.write(final_audio_file, y_processed, sr)

        # Convert back to MP3 using FFmpeg from imageio_ffmpeg
        final_mp3 = "processed_audio.mp3"
        subprocess.run(
            [ffmpeg_path, "-i", final_audio_file, final_mp3, "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Provide download link
        with open(final_mp3, "rb") as file:
            st.download_button(
                label="Download Processed Audio",
                data=file,
                file_name="processed_audio.mp3",
                mime="audio/mpeg",
            )

        # Clean up temporary files
        if os.path.exists(audio_file):
            os.remove(audio_file)
        if os.path.exists(intermediate_file):
            os.remove(intermediate_file)
        if os.path.exists(final_audio_file):
            os.remove(final_audio_file)
        if os.path.exists(final_mp3):
            os.remove(final_mp3)

    except Exception as e:
        st.error(f"An error occurred: {e}")
