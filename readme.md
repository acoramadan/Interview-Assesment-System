# Cheating Detection (MediaPipe) + ASR Pipeline (VAD → Diarization → Faster-Whisper)

## Testing Output
<div style="display:flex; gap:10px; justify-content:center; align-items:center; overflow-x:auto; padding:8px 0;">
  <img src="/assets/1.gif" alt="Demo 1" height="140">
  <img src="/assets/2.gif" alt="Demo 2" height="140">
  <img src="/assets/3.gif" alt="Demo 3" height="140">
  <img src="/assets/4.gif" alt="Demo 4" height="140">
  <img src="/assets/5.gif" alt="Demo 5" height="140">
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

Tujuan Singkat
Transkripsi audio atau audio hasil ekstrak video dengan output: transkrip, speaker, timestamp, dan confidence. Bisa word-level.

Fitur Utama
VAD (Silero): memilih bagian bersuara agar hemat komputasi.

Speaker Diarization (pyannote): segmentasi per pembicara.

ASR (Faster-Whisper): transkripsi per segmen, mendukung language, beam_size, dan word timestamps.

Confidence per segmen dari avg_logprob dan no_speech_prob.

Word-level (opsional): kata, timestamp global, probability.

Komponen: VAD, Diarization, Transcriber, ASRPipeline.

Instalasi Cepat (ASR)
bash
Salin kode
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --upgrade pip
pip install torch                   # pilih wheel CUDA sesuai driver jika ada GPU
pip install pyannote.audio faster-whisper soundfile fastapi uvicorn

# opsional: ffmpeg untuk ekstraksi/resampling audio dari video
Cara Pakai Tingkat Tinggi
Ekstrak audio dari video (opsional, via ffmpeg).

Jalankan VAD untuk mendeteksi bagian bersuara.

Jalankan diarization untuk memisahkan pembicara.

Transkripsi setiap segmen dengan Faster-Whisper.

Gabungkan hasil: text, speaker, start, end, confidence (opsional: word-level).

Integrasi Vision + ASR (Opsional)
Jalankan keduanya secara paralel atau berurutan.

Samakan referensi waktu. Gunakan segmen vision (EYES_OFF/EYES_MOVING) sebagai konteks ketika menganalisis reliabilitas transkrip atau menandai event kecurangan multimodal.

Catatan Konfigurasi Singkat
Atur sensitivitas mata: turunkan GAZE_SPEED_THR atau GAZE_DELTA_X/Y bila perlu.

Kurangi salah deteksi karena gelengan: naikkan HEAD_STATIC_VEL_THR atau longgarkan FACING_YAW_DEG/PITCH_DEG.

Stabilkan sinyal: kecilkan EMA_ALPHA untuk pergerakan yang lebih halus.

Prasyarat Umum
Python 3.9–3.11

Kamera terpasang untuk vision

GPU opsional untuk percepatan torch dan Faster-Whisper