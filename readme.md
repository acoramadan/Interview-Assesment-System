# Cheating Detection (MediaPipe) + ASR Pipeline (VAD → Diarization → Faster-Whisper)

## Testing Output
<div style="display:flex; gap:10px; justify-content:center; align-items:center; overflow-x:auto; padding:8px 0;">
  <img src="/assets/1.gif" alt="Demo 1" height="140">
  <img src="/assets/2.gif" alt="Demo 2" height="140">
  <img src="/assets/3.gif" alt="Demo 3" height="140">
  <img src="/assets/5.gif" alt="Demo 4" height="140">
</div>

---
# Arsitektur Model
<img src="/assets/arsitektur.png" title="Model architecture">

## 1. Cheating Detection with MediaPipe (Head-tolerant, Eye-priority)

### Tujuan Singkat
Deteksi indikasi tidak fokus atau kecurangan dari kamera dengan menekankan pergerakan bola mata (eye-gaze) dan memberi toleransi pada gerakan kepala yang ekspresif.

### Cara Kerja Singkat
1. Ambil frame dari kamera pada target FPS.
2. Deteksi wajah (MediaPipe FaceDetection) untuk bounding box.
3. Estimasi landmark halus + iris (MediaPipe FaceMesh).
4. Estimasi baseline pose kepala (otomatis) untuk bias pengguna
5. Hitung:
   - Pose kepala (yaw, pitch, roll) via solvePnP.
   - Keterbukaan mata dan posisi iris → gaze (gx, gy).
   - Kecepatan kepala dan kecepatan gaze.
7. Tentukan status dan alasan:
   - Fokus atau tidak (dengan hysteresis).
   - Alasan prioritas: MULTIPLE_FACES → OUT_OF_FRAME → EYES_OFF → EYES_MOVING → HEAD_POSE_OFF.
8. Tulis segmen ke CSV/JSON saat alasan berubah.

### Komponen Utama
- PeopleDetector: mendeteksi jumlah orang per frame menggunakan YOLOv8 Pose, dilengkapi filter confidence, luas bbox, validasi skeleton, dan smoothing temporal.
- EyeGazeEstimator: estimasi arah pandangan mata berbasis MediaPipe FaceMesh + iris landmarks, menghasilkan vektor gaze (gx, gy) dan label CENTER/LEFT/RIGHT/UP/DOWN/BLINK.
- HeadPoseEstimator: estimasi pose kepala (yaw, pitch, roll) menggunakan solvePnP, dengan smoothing dan koreksi baseline otomatis.
- HybridCalibration: estimasi baseline yaw dan pitch dari frame awal secara otomatis (tanpa interaksi pengguna) untuk koreksi bias posisi kamera.
- SegmentLogger: log segmen ke cheat_outputs/<prefix>_segments.csv dan .json.

### Parameter Kunci (disetel agar pro-eye, toleran kepala)
- FACING_YAW_THRESHOLD=25, HEAD_PITCH_THRESHOLD=20
- EYE_CHEAT_MIN_DURATION = 1.0s, HEAD_CHEAT_MIN_DURATION = 1.0s
- PEOPLE_MIN_DURATION = 0.3s
- BLINK_EAR = 0.20
- GAZE_LEFT/RIGHT/UP/DOWN_TH ≈ ±0.12
- Head smoothing ALPHA = 0.7
- Person count smoothing strength = 0.6
- CALIBRATION_LIMIT = 40 frames (~1 detik)

### Output
- Overlay kamera: bbox, status, yaw/pitch, head_vel, gaze, kecepatan gaze, dan CHEATING:REASON saat aktif.
- CSV kolom: track_id, reason, start_ts, end_ts, duration_sec, start_hhmmss, end_hhmmss
- JSON: metadata sesi dan daftar segmen.

# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --upgrade pip
pip install opencv-python numpy mediapipe
Menjalankan (Vision)
bash
Salin kode
python main.py
Tekan Esc untuk keluar. Hasil segmen tersimpan di cheat_outputs/.



# 2. Automatic Speech Recognition (ASR) Pipeline  
*(Interview Assessment System – ASR Version)*

## Tujuan Singkat
Melakukan transkripsi wawancara secara **tersegmentasi per pembicara** dari video interview, sehingga jawaban kandidat dapat dianalisis dan dinilai secara otomatis berdasarkan **rubric pertanyaan**.

Pipeline ini berfokus pada **akurasi transkrip + pemisahan pembicara (speaker diarization)** dan menghasilkan output terstruktur (JSON) yang siap dikonsumsi oleh modul penilaian (LLM / Gemini API).

---

## Arsitektur (Sesuai Diagram)
**Video Dataset → Audio Extraction → ASR & Diarization → Transcript Segments JSON → Assessment → Final Report**

ASR berada dalam satu blok besar dan terhubung langsung dengan:
- **Video Dataset** (input)
- **Question Rubric (YAML)** (konteks penilaian)
- **Gemini API** (penilaian jawaban)
- **Final Report** (output akhir)

---

## Cara Kerja Singkat
1. **Input Video**
   - Video interview diambil dari *Video Dataset*.

2. **Extract Audio**
   - Audio diekstrak dari video:
     - Sample rate: **16 kHz**
     - Channel: **mono**
   - Output: `file.wav`

3. **Speaker Diarization + VAD (Pyannote)**
   - `pyannote.audio` digunakan untuk:
     - Voice Activity Detection (VAD)
     - Speaker diarization
   - Output berupa segmen waktu dengan label speaker:
     ```
     [start_time, end_time, speaker_id]
     ```

4. **Automatic Speech Recognition (Whisper Large)**
   - Model: **Whisper Large**
   - Input: audio + segmen diarization
   - Output:
     - Transkrip teks
     - Timestamp per segmen
     - Speaker-aware transcription

5. **Transcript Segments JSON**
   - Semua hasil ASR disatukan dalam format JSON terstruktur
   - Menjadi sumber utama untuk penilaian jawaban

6. **Assessment dengan Gemini API**
   - Transcript + Question Rubric (YAML) dikirim ke Gemini API
   - Sistem menilai:
     - Relevansi jawaban
     - Kelengkapan
     - Kejelasan
     - Konsistensi antar segmen

7. **Final Report**
   - Menggabungkan:
     - Skor jawaban (ASR)
     - Hasil cheating detection (Computer Vision)
   - Output akhir dalam bentuk laporan evaluasi interview

---

## Komponen Utama

### 1. AudioExtractor
- Mengambil audio dari video
- Normalisasi:
  - 16kHz
  - mono
- Output: `file.wav`

---

### 2. DiarizationEngine
- Library: `pyannote.audio`
- Fungsi:
  - Voice Activity Detection
  - Speaker diarization
- Output:
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 3.21,
    "end": 8.95
  }
]
```

### 3.Hasil Uji dengan OpenLSR
<div> </div>
<img src="assets/evalasr.png" title="Output ASR">
<div> </div>


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

GPU opsional untuk percepatan torch dan Faster-Whisper
