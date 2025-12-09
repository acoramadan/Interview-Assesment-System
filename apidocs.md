# Video Upload API with FastAPI

API sederhana untuk upload dan manajemen video menggunakan FastAPI.

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Menjalankan Server

```bash
python main.py
```

atau

```bash
uvicorn main:app --reload
```

Server akan berjalan di: http://localhost:8000

## API Documentation

Setelah server berjalan, buka:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### 1. Upload Video
- **URL**: `/upload`
- **Method**: `POST`
- **Body**: Form-data dengan key `file`
- **Response**: Informasi file yang diupload

### 2. List Videos
- **URL**: `/videos`
- **Method**: `GET`
- **Response**: List semua video yang sudah diupload

### 3. Delete Video
- **URL**: `/videos/{filename}`
- **Method**: `DELETE`
- **Response**: Konfirmasi penghapusan

### 4. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**: Status server

## Format Video yang Didukung

- MP4
- AVI
- MOV
- MKV
- WEBM
- FLV
- WMV

## Folder Upload

Video akan disimpan di folder: `uploaded_videos/`

## Testing dengan cURL

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/video.mp4"
```
