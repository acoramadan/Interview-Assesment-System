import torch
import os
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
AUDIO_NAME = 'interview_question_5.wav'
RESULT_PATH = '../result/asr_output/'

print("Loading VAD model...\n")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir = "snakers4/silero-vad",
    model = VAD_MODEL_NAME,
    force_reload = False,
    trust_repo = True,
)
(get_speech_timestamps, _, read_audio, _, _) = vad_utils

print("Loading Diarization model...\n")
pyannote = Pipeline.from_pretrained(PYANNOTE_VERSION, token=HG_TOKEN)
whisper = WhisperModel(WHISPER_MODEL_SIZE, device=get_device(), compute_type="float16")

vad = VAD(vad_model=vad_model, sampling_rate=SR)
diar = Diarization(pyannote_model=pyannote, sampling_rate=SR)
transcribe = Transcribe(model=whisper, sampling_rate=SR)
pipe = ASRPipeline(vad=vad, diar=diar, transcribe=transcribe)

print("Reading audio file...\n")
audio, sr_read = sf.read(os.path.join(AUDIO_PATH, AUDIO_NAME), dtype='float32')
assert sr_read == SR, f"Sample rate mismatch: expected {SR}, got {sr_read}"

print("Running ASR pipeline...\n")
hyp, segments = pipe.process(
    audio=audio,
    get_speech_timestamps=get_speech_timestamps,
    language='en',
    return_words=True,
)
full_scripts = " ".join([seg['text'] for seg in segments])

output_data = {
    "full_scripts": full_scripts,
    "segments": segments
}

with open(os.path.join(RESULT_PATH, AUDIO_NAME.replace('.wav', '_asr_output.json')), 'w') as f:
    json.dump(output_data, f, indent=4)
print("\nASR pipeline completed. Results saved.\n")