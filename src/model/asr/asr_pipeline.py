from .voice_detection import VAD
from .diarization import Diarization
from .transcribe import Transcribe
from typing import Tuple, List, Dict
import numpy as np

class ASRPipeline:
    def __init__(self, vad: VAD, diar: Diarization, transcribe: Transcribe):
        self.vad = vad
        self.diar = diar
        self.transcribe = transcribe
    
    def process(
            self,
            audio: np.ndarray,
            get_speech_timestamps,
            language: str = 'en',
            return_words: bool = True,
            min_speech: int = 250,
            min_silence: int = 200,
    ) -> Tuple[str, List[Dict]]:
        
        vad_sec = self.vad.timestamp(
            wav = audio,
            get_speech_timestamps = get_speech_timestamps,
            min_speech = min_speech,
            min_silence = min_silence,
        )
        
        diar_segments = self.diar.run(vad_sec, audio)

        asr_segments = self.transcribe.transcribe_segment(
            diar_segments=diar_segments, language=language, audio=audio, return_words=return_words
        )
        full_text = " ".join(seg['text'] for seg in asr_segments).strip()
        return full_text, asr_segments