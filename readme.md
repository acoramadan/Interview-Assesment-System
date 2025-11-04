
# Cheating detection with media pipe
**Testing output**
<div style="display:flex; gap:10px; justify-content:center; align-items:center; overflow-x:auto; padding:8px 0;">
  <img src="/assets/1.gif" alt="Demo 1" height="140">
  <img src="/assets/2.gif" alt="Demo 2" height="140">
  <img src="/assets/3.gif" alt="Demo 3" height="140">
  <img src="/assets/4.gif" alt="Demo 4" height="140">
  <img src="/assets/5.gif" alt="Demo 5" height="140">
</div>

# ASR Pipeline – VAD → Diarization → Faster-Whisper
**Arsitektur model**
<img src="/assets/image.png"  title="Model architechture">

**Result of testing with openLSR dataset**
<img src="/assets/outputasr.png"  title="Model architechture">

Pipeline inference **speaker-aware** untuk audio (atau audio hasil ekstraksi video) dengan output:
**transkrip + speaker + timestamp + confidence** (opsional: word-level).
Didesain modular (OOP) sehingga bisa dipakai sebagai **library (web API)** atau **CLI**.

## Fitur Utama
- **VAD (Silero)**: deteksi bagian ber-suara → hemat komputasi.
- **Speaker Diarization (pyannote)**: segmentasi per-pembicara.
- **ASR (Faster-Whisper)**: transkripsi tiap segmen (dukung `language`, `beam_size`, word timestamps).
- **Confidence score** per segmen: dari `avg_logprob` & `no_speech_prob`.
- **Word-level (opsional)**: kata, timestamp global, probability.


> - `VAD`, `Diarization`, `Transcriber`, dan `ASRPipeline`.

---

## Prasyarat & Dependensi

- Python 3.9–3.11
- `torch` (GPU opsional, direkomendasikan)
- `pyannote.audio` (model: `pyannote/speaker-diarization-community-1`)
- `faster-whisper`
- `soundfile`
- (Opsional) `ffmpeg` untuk resampling / ekstrak audio dari video

### Instalasi Cepat

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install --upgrade pip

# inti
pip install torch                       # atau wheel CUDA sesuai driver Anda
pip install pyannote.audio faster-whisper soundfile fastapi uvicorn

