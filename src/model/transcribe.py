from typing import List, Dict
from .utils import _as_float32_contiguous, set_confidence
import numpy as np

class Transcribe:

    def __init__(self, model, sampling_rate=16000):
        self.model = model
        self.sr = sampling_rate

    def transcribe_segment(
            self,
            diar_segments: List[Dict],
            audio: np.ndarray,
            language: str = 'en',
            min_seg_sec: float = 0.4,
            beam_size: int = 5,
            return_words: bool = True,
    ):
        results = []
        for seg in diar_segments:
            dur = seg['end'] - seg['start']
            if dur < min_seg_sec:
                continue
            
            s_samp = int(seg['start'] * self.sr)
            e_samp = int(seg['end'] * self.sr)
            chunk = _as_float32_contiguous(audio[s_samp:e_samp])

            kw = dict(beam_size=beam_size, word_timestamps=return_words)
            if language:
                kw['language'] = language

            segs, info = self.model.transcribe(chunk, **kw)
            text = " ".join([s.text for s in segs]).strip()

            if segs:
                has_lp = any(getattr(s, 'avg_logprob', None) is not None for s in segs)
                avg_lp = float(np.mean([s.avg_logprob for s in segs if s.avg_logprob is not None])) if has_lp else -5.0
                has_nsp = any(getattr(s, "no_speech_prob", None) is not None for s in segs)
                nsp = float(np.mean([s.no_speech_prob for s in segs if s.no_speech_prob is not None])) if has_nsp else 0.0
            
            else:
                avg_lp, nsp = -5.0, 0.0
            
            conf = set_confidence(avg_lp, nsp)

            item = {
                "start": seg['start'],
                "end": seg['end'],
                "speaker": seg["speaker"],
                "text": text,
                "confidence": conf,
            }
            if return_words:
                words_out = []
                for s in segs:
                    if getattr(s, "words", None):
                        for w in s.words:
                            prob = getattr(w, "probability", None)
                            words_out.append({
                                "word": w.word,
                                "start": (w.start or 0.0) + seg['start'],
                                "end": (w.end or 0.0) + seg['start'],
                                'probability': float(prob) if prob is not None else None
                            })
            

            item["words"] = words_out
            results.append(item)
        return results