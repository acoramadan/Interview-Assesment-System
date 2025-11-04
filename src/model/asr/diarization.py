import numpy as np
import torch
from typing import List, Dict
from .utils import _as_float32_contiguous

class Diarization:
    def __init__(self, pyannote_model, sampling_rate: int = 16000):
        self.model = pyannote_model
        self.sr = sampling_rate
    
    def run(self, vad_segments: List[Dict], audio: np.ndarray) -> List[Dict]:
        all_diar = []
        for seg in vad_segments:
            st, en = seg['start'], seg['end']
            s_samp, e_samp = int(st * self.sr), int(en * self.sr)
            chunk = _as_float32_contiguous(audio[s_samp:e_samp])
            wav_tensor = torch.from_numpy(chunk).unsqueeze(0)

            with torch.no_grad():
                out = self.model({'waveform': wav_tensor, 'sample_rate': self.sr})

                for turn, speaker in out.speaker_diarization:
                    all_diar.append({
                        'start': st + turn.start,
                        'end': st + turn.end,
                        'speaker': speaker
                    })
        
        return all_diar