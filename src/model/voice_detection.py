import torch
import numpy as np
from typing import Tuple, List, Dict

class VAD:
    def __init__(self, vad_model, sampling_rate=16000):
        self.vad_model = vad_model
        self.sr = sampling_rate
    
    @staticmethod
    def _convert_to_sec(start_sample: int, end_sample: int, sr=16000) -> Tuple[float, float]:
        return start_sample /sr, end_sample / sr
    
    @staticmethod
    def _merge_segments(segments: List[Dict], merge_gap: float=0.2, min_len: float = 0.25) -> List[Dict]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda x: x['start'])
        merged = []
        cur = segments[0].copy()

        for seg in segments[1:]:
            gap = seg['start'] - cur['end']

            if gap <= merge_gap:
                cur['end'] = max(cur['end'], seg['end'])
            else:
                if (cur['end'] - cur['start']) >= min_len:
                    merged.append(cur)
                cur = seg.copy()
        
        if (cur['end'] - cur['start']) >= min_len:
            merged.append(cur)
        
        return merged
    
    def timestamp(
            self,
            wav: np.ndarray,
            get_speech_timestamps,
            min_speech: int = 250,
            min_silence: int = 200,
    ) -> List[Dict]:
        
        raw = get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate = self.sr,
            min_speech_duration_ms = min_speech,
            min_silence_duration_ms = min_silence,
        )

        for i in range(len(raw)):
            raw[i]['start'], raw[i]['end'] = self._convert_to_sec(raw[i]['start'], raw[i]['end'], sr=self.sr)
            
        return self._merge_segments(raw)


