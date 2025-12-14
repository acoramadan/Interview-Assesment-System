
# Interview Assessment System  
**Cheating Detection (Computer Vision) + ASR Pipeline (VAD → Diarization → Faster-Whisper)**

Sistem ini merupakan **working prototype (MVP)** untuk membantu proses evaluasi interview video dengan mengombinasikan **analisis verbal (speech-to-text)** dan **indikator non-verbal (computer vision)**.

---

## Demo Output
Lihat folder `assets/` untuk contoh GIF hasil pengujian sistem.

---

## 1. Cheating Detection – Computer Vision (MediaPipe)

### Tujuan
Mendeteksi **indikasi ketidakfokusan atau potensi kecurangan** dari kamera dengan:
- Prioritas pada **pergerakan mata (eye-gaze)**
- Toleransi terhadap **gerakan kepala yang ekspresif**

Modul ini **tidak mengambil keputusan otomatis**, melainkan menghasilkan **indikator pendukung** untuk proses evaluasi.

### Cara Kerja Singkat
1. Mengambil frame dari video/kamera pada FPS tertentu.  
2. Mendeteksi wajah menggunakan *MediaPipe FaceDetection*.  
3. Mengestimasi landmark wajah dan iris menggunakan *MediaPipe FaceMesh*.  
4. Mengestimasi baseline pose kepala secara otomatis (tanpa interaksi pengguna).  
5. Menghitung pose kepala, arah pandangan mata, serta kecepatan gerak.  
6. Menentukan status fokus dengan *rule-based logic* dan hysteresis.  
7. Mencatat segmen ke CSV/JSON ketika status berubah.

### Komponen Utama
- `PeopleDetector`: deteksi jumlah orang per frame menggunakan **YOLOv8 Pose**.  
- `EyeGazeEstimator`: estimasi arah pandangan mata berbasis **MediaPipe FaceMesh + iris landmarks**.  
- `HeadPoseEstimator`: estimasi pose kepala (yaw, pitch, roll) menggunakan **solvePnP**.  
- `HybridCalibration`: estimasi baseline yaw dan pitch dari frame awal secara otomatis.  
- `SegmentLogger`: pencatatan segmen event ke file CSV dan JSON.

### Parameter Kunci
- `HEAD_YAW_THRESHOLD = 25°`, `HEAD_PITCH_THRESHOLD = 20°`  
- `EYE_CHEAT_MIN_DURATION = 1.0s`  
- `HEAD_CHEAT_MIN_DURATION = 1.0s`  
- `PEOPLE_MIN_DURATION = 0.3s`  
- `BLINK_EAR = 0.20`  
- `GAZE_LEFT / RIGHT / UP / DOWN_TH ≈ ±0.12`  
- `CALIBRATION_LIMIT = 40 frames (~1 detik)`

### Output
- Overlay kamera: bounding box, status, yaw/pitch, gaze, dan alasan event.  
- CSV: segmen event per durasi.  
- JSON: metadata sesi dan daftar segmen.

---

## 2. ASR Pipeline – VAD → Diarization → Faster-Whisper

### Tujuan
Menyediakan **transkripsi interview berbasis audio** yang *speaker-aware*, memiliki timestamp, dan siap diintegrasikan ke sistem evaluasi.

### Alur Pipeline
1. (Opsional) Ekstraksi audio dari video menggunakan FFmpeg.  
2. **Voice Activity Detection (VAD)** untuk mendeteksi segmen bersuara.  
3. **Diarization** untuk memisahkan pembicara.  
4. Transkripsi tiap segmen menggunakan **Faster-Whisper**.  
5. Penggabungan hasil: teks, speaker, waktu mulai/akhir, confidence (opsional).

### Output ASR
- Transkrip teks  
- Label speaker  
- Timestamp per segmen  
- Confidence (opsional)

---

## Integrasi Vision + ASR (Opsional)
- Vision dan ASR dapat dijalankan secara paralel atau berurutan.  
- Segmen Vision digunakan sebagai konteks tambahan dalam analisis interview.

---

## Instalasi Singkat
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux / macOS

pip install --upgrade pip
pip install opencv-python numpy mediapipe torch pyannote.audio faster-whisper soundfile
```

---

## Catatan
- Sistem ini **tidak menggantikan assessor**.  
- Output bersifat **indikator pendukung**, bukan keputusan final.  
- Dirancang sebagai **MVP modular**.

---

## Prasyarat
- Python 3.9 – 3.11  
- Kamera (untuk modul Vision)  
- GPU opsional (untuk ASR)
