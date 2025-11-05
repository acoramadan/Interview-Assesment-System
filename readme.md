# Cheating Detection (MediaPipe) + ASR Pipeline (VAD → Diarization → Faster-Whisper)

## Testing Output
<div style="display:flex; gap:10px; justify-content:center; align-items:center; overflow-x:auto; padding:8px 0;">
  <img src="/assets/1.gif" alt="Demo 1" height="140">
  <img src="/assets/2.gif" alt="Demo 2" height="140">
  <img src="/assets/3.gif" alt="Demo 3" height="140">
  <img src="/assets/4.gif" alt="Demo 4" height="140">
</div>

---

## 1. Cheating Detection with MediaPipe (Head-tolerant, Eye-priority)

### Tujuan Singkat
Deteksi indikasi tidak fokus atau kecurangan dari kamera dengan menekankan pergerakan bola mata (eye-gaze) dan memberi toleransi pada gerakan kepala yang ekspresif.

### Cara Kerja Singkat
1. Ambil frame dari kamera pada target FPS.
2. Deteksi wajah (MediaPipe FaceDetection) untuk bounding box.
3. Estimasi landmark halus + iris (MediaPipe FaceMesh).
4. Hitung:
   - Pose kepala (yaw, pitch, roll) via solvePnP.
   - Keterbukaan mata dan posisi iris → gaze (gx, gy).
   - Kecepatan kepala dan kecepatan gaze.
5. Kalibrasi 2 detik untuk bias pose dan pusat gaze per pengguna.
6. Tentukan status dan alasan:
   - Fokus atau tidak (dengan hysteresis).
   - Alasan prioritas: MULTIPLE_FACES → OUT_OF_FRAME → EYES_OFF → EYES_MOVING → HEAD_POSE_OFF.
7. Tulis segmen ke CSV/JSON saat alasan berubah.

### Komponen Utama
- `Track`: status per wajah (pose tersmooth, head velocity, gaze, timers, flags).
- `Tracker`: asosiasi bbox antar frame (IOU) dan pembersihan track hilang.
- `Calib`: bias yaw/pitch dan pusat gaze dari median sampel awal.
- `SegmentLogger`: log segmen ke `cheat_outputs/<prefix>_segments.csv` dan `.json`.

### Parameter Kunci (disetel agar pro-eye, toleran kepala)
- `FACING_YAW_DEG=30`, `FACING_PITCH_DEG=20`
- `MIN_FACING_FOR_GAZE=16`
- `HYSTERESIS_FRAMES=8`
- `CHEAT_MIN_EYESMOV_SEC=0.6`, `CHEAT_MIN_EYESOFF_SEC=0.8`
- `GAZE_SPEED_THR=0.25`, `GAZE_SPEED_STATIC_HEAD_THR=25`
- `HEAD_STATIC_VEL_THR=35 deg/s`

### Output
- Overlay kamera: bbox, status, yaw/pitch, head_vel, gaze, kecepatan gaze, dan CHEATING:REASON saat aktif.
- CSV kolom: `track_id, reason, start_ts, end_ts, duration_sec, start_hhmmss, end_hhmmss`
- JSON: metadata sesi dan daftar segmen.

### Instalasi Cepat (Vision)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --upgrade pip
pip install opencv-python numpy mediapipe
Menjalankan (Vision)
bash
Salin kode
python main.py
Tekan Esc untuk keluar. Hasil segmen tersimpan di cheat_outputs/.
```

#### 2. ASR Pipeline – VAD → Diarization → Faster-Whisper

Arsitektur Model
<img src="/assets/image.png" title="Model architecture">

Hasil Uji dengan OpenLSR
<div> </div>
<img src="/assets/outputasr.png" title="Output ASR">
<div> </div>

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