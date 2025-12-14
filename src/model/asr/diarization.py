import numpy as np
import torch
from typing import List, Dict

class Diarization:
    def __init__(self, pyannote_model, sampling_rate: int = 16000):
        self.model = pyannote_model
        self.sr = sampling_rate
    
    def run(self, audio: np.ndarray) -> List[Dict]:
        # Convert audio ke tensor
        wav_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

        # Jalankan model diarization
        with torch.no_grad():
            diar_output = self.model({
                "waveform": wav_tensor,
                "sample_rate": self.sr
            })

        # ======== PERBAIKAN PENTING =========
        # pyannote 3.1 → wrapper object
        # Annotation ada di: diar_output.speaker_diarization
        # ====================================
        if not hasattr(diar_output, "speaker_diarization"):
            raise RuntimeError(
                f"Diarization output tidak memiliki 'speaker_diarization'. "
                f"Tipe: {type(diar_output)}"
            )

        annotation = diar_output.speaker_diarization

        all_diar = []

        # Iterasi segment diarization
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            all_diar.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": speaker
            })

        return all_diar
