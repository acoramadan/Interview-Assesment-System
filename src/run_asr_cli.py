import torch
import os
import argparse
import soundfile as sf
import warnings
import numpy as np
import json
import re
import glob
import soundfile as sf
from dotenv import load_dotenv
from jiwer import wer, cer, Compose, ToLowerCase, ToUpperCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from model.asr.utils import get_device
from model.asr.asr_pipeline import ASRPipeline
from model.asr.voice_detection import VAD
from model.asr.diarization import Diarization
from model.asr.transcribe import Transcribe

warnings.filterwarnings("ignore")

load_dotenv()
SR = 16000
AUDIO_PATH = '../data/audio/'
HG_TOKEN = os.getenv("HG_TOKEN")
WHISPER_MODEL_SIZE = 'medium.en'
PYANNOTE_VERSION = 'pyannote/speaker-diarization-community-1'
VAD_MODEL_NAME = 'silero_vad'
AUDIO_NAME = 'tes.wav'
RESULT_PATH = '../result/'

def main():
    parser = argparse.ArgumentParser(description="ASR pipeline: VAD + Diarization + Faster-Whisper")
    parser.add_argument('--audio', type=str, required=True, default=os.path.join(AUDIO_PATH, AUDIO_NAME), help='Path audio (.wav/.flac) 16k mono')
    parser.add_argument('--output', type=str, required=False, default=RESULT_PATH, help='Path to save the ASR output JSON')
    parser.add_argument("--return_words", action="store_true", help="Whether to return word-level timestamps")
    parser.add_argument("--min_speech_ms", type=int, default=250)
    parser.add_argument("--min_silence_ms", type=int, default=200)

    args = parser.parse_args()

    print("Loading VAD model...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir = "snakers4/silero-vad",
        model = VAD_MODEL_NAME,
        force_reload = False,
        trust_repo = True,
    )
    (get_speech_timestamps, _, read_audio, _, _) = vad_utils

    print("Loading Diarization model...")
    pyannote = Pipeline.from_pretrained(PYANNOTE_VERSION, token=HG_TOKEN)
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device=get_device(), compute_type="float16")

    vad = VAD(vad_model=vad_model, sampling_rate=SR)
    diar = Diarization(pyannote_model=pyannote, sampling_rate=SR)
    transcribe = Transcribe(model=whisper, sampling_rate=SR)
    pipe = ASRPipeline(vad=vad, diar=diar, transcribe=transcribe)

    print("Reading audio file...")
    audio, sr_read = sf.read(args.input, dtype="float32")
    print("Audio file read successfully.")

    if sr_read != SR:
        raise ValueError(f"Sample rate harus {SR}, dapat {sr_read}. (Resample dulu ke 16k mono)")
    
    hyp, segments = pipe.process(
        audio=audio,
        get_speech_timestamps=get_speech_timestamps,
        return_words=args.return_words,
        min_speech=args.min_speech_ms,
        min_silence=args.min_silence_ms,
    )
    print("Transcription Results:\n")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, os.path.basename(args.audio).replace('.wav', '_asr_output.json')), 'w') as f:
        json.dump(segments, f, indent=4)
    
    print(f"ASR pipeline completed. Results saved to {args.output}\n")

if __name__ == "__main__":
    main()
