from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from datetime import datetime
from pathlib import Path
import uvicorn
import subprocess
import torch
import soundfile as sf
import warnings
import json
import cv2
import mediapipe as mp
import numpy as np
import yaml
import time
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from google import genai
from google.genai import types
from google.genai.errors import ServerError

from model.asr.asr_pipeline import ASRPipeline
from model.asr.diarization import Diarization
from model.asr.transcribe import Transcribe

from run_comvis_model import run_full_pipeline, postprocess_and_export

warnings.filterwarnings("ignore")
load_dotenv()

app = FastAPI(title="Video Upload API with ASR, Cheating Detection & Assessment", version="5.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "../data/uploaded_videos"
AUDIO_FOLDER = "../data/extracted_audio"
ASR_RESULT_FOLDER = "../result/asr_results"
CHEAT_RESULT_FOLDER = "../result/cheat_results"
REASONING_RESULT_FOLDER = "../result/reasoning_results"
CONFIG_PATH = "../conf.yaml"
SR = 16000
HG_TOKEN = os.getenv("HG_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHISPER_MODEL_SIZE = 'large-v3'
PYANNOTE_VERSION = 'pyannote/speaker-diarization-community-1'

for folder in [UPLOAD_FOLDER, AUDIO_FOLDER, ASR_RESULT_FOLDER, CHEAT_RESULT_FOLDER, REASONING_RESULT_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

models_loaded = False
vad_model = None
vad_utils = None
pyannote = None
whisper = None

def load_models():
    """Load all ASR models at startup"""
    global models_loaded, vad_model, vad_utils, pyannote, whisper

    if models_loaded:
        return

    try:
        print("Loading Diarization model...")
        pyannote = Pipeline.from_pretrained(PYANNOTE_VERSION, token=HG_TOKEN)

        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        whisper = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)

        models_loaded = True
        print("All ASR models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def load_config(path: str = CONFIG_PATH) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return None

def assess_answer(qid: str, candidate_answer: str) -> dict:
    """Assess answer using Gemini API"""
    try:
        cfg = load_config()
        if cfg is None:
            return {
                "success": False,
                "error": "Config file not found"
            }

        model_cfg = cfg["model"]
        rubric = cfg["rubric"].get(qid)

        if rubric is None:
            return {
                "success": False,
                "error": f"Question {qid} not found in rubric"
            }

        prompt_cfg = cfg["prompts"]["assesor_single"]

        user_prompt = prompt_cfg["template"].format(
            question=rubric["question"],
            lvl4=rubric["scale"]["4"],
            lvl3=rubric["scale"]["3"],
            lvl2=rubric["scale"]["2"],
            lvl1=rubric["scale"]["1"],
            lvl0=rubric["scale"]["0"],
            answer=candidate_answer.strip(),
        )

        client = genai.Client(api_key=GEMINI_API_KEY)

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ]

        tools = []
        if model_cfg.get("use_google_search", False):
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        gen_config = types.GenerateContentConfig(
            system_instruction=prompt_cfg["system"],
            temperature=model_cfg.get("temperature", 0.0),
            max_output_tokens=model_cfg.get("max_output_tokens", 2048),
            response_mime_type=model_cfg.get("response_mime_type", "application/json"),
            thinking_config=types.ThinkingConfig(
                thinking_budget=model_cfg.get("thinking_budget", -1)
            ),
            tools=tools or None,
        )

        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model_cfg["model_name"],
                    contents=contents,
                    config=gen_config,
                )
                break
            except ServerError as e:
                if e.code == 503 and attempt < 2:
                    wait_time = (2 ** attempt) * 2
                    print(f"ServerError 503 encountered. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e

        if resp.text is None:
            texts = []
            for cand in resp.candidates:
                for part in cand.content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    if getattr(part, "text", None):
                        texts.append(part.text)
            joined = "\n".join(texts)
            if not joined:
                raise RuntimeError("No text in response. Check prompt or safety blocks.")
            result = json.loads(joined)
        else:
            result = json.loads(resp.text)

        return {
            "success": True,
            "score": result.get("score"),
            "reason": result.get("reason"),
            "matched_level": result.get("matched_level")
        }

    except Exception as e:
        print(f"Assessment error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

def convert_hhmmss_to_mmss(hhmmss_str: str) -> str:
    """Convert HH:MM:SS.mmm to MM:SS format"""
    try:
        parts = hhmmss_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total_seconds = hours * 3600 + minutes * 60 + int(seconds)
            mins = total_seconds // 60
            secs = total_seconds % 60
            return f"{mins:02d}:{secs:02d}"
        return hhmmss_str
    except:
        return hhmmss_str

def is_allowed_file(filename: str) -> bool:
    """Check if file is valid video"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using FFmpeg"""
    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(SR),
            '-ac', '1',
            '-y',
            audio_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )

        return result.returncode == 0

    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

def process_asr(audio_path: str, language: str) -> dict:
    """Process audio with ASR pipeline"""
    try:
        audio, sr_read = sf.read(audio_path, dtype='float32')
        assert sr_read == SR, f"Sample rate mismatch: expected {SR}, got {sr_read}"

        diar = Diarization(pyannote_model=pyannote, sampling_rate=SR)
        transcribe = Transcribe(model=whisper, sampling_rate=SR)
        pipe = ASRPipeline(diar=diar, transcribe=transcribe)

        hyp, segments = pipe.process(
            audio=audio,
            language=language,
            return_words=True,
        )

        full_scripts = " ".join([seg['text'] for seg in segments])
        speakers = set([seg['speaker'] for seg in segments])
        total_speakers = len(speakers)
        total_duration = sum([seg['end'] - seg['start'] for seg in segments])
        

        return {
            "success": True,
            "full_scripts": full_scripts,
            "segments": segments,
            "total_speakers": total_speakers,
            "speaker_list": list(speakers),
            "total_duration_seconds": round(total_duration, 2)
        }

    except Exception as e:
        print(f"ASR processing error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def process_cheating_detection(video_path: str) -> dict:
    """Wrapper untuk menjalankan pipeline comvis baru (Arya)
       tetapi outputnya mengikuti format pipeline lama temanmu.
    """
    try:
        # 1. Jalankan pipeline comvis kamu
        eye_frames, head_frames, people_frames, fps, total_frames = run_full_pipeline(video_path)
        output, json_path, csv_path = postprocess_and_export(
            eye_frames, head_frames, people_frames, fps, total_frames
        )

        events = output.get("segments", [])

        # 2. Bangun reason_counts dan reason_details mirip pipeline temanmu
        reason_counts = {}
        reason_details = {}

        for event in events:
            reason = event.get("reason", "UNKNOWN")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if reason not in reason_details:
                reason_details[reason] = []

            # Konversi hh:mm:ss.mmm → mm:ss
            def to_mmss(hhmmss):
                h, m, s = hhmmss.split(":")
                return f"{int(m) + int(h)*60}:{s.split('.')[0]}"

            start_mmss = to_mmss(event["start_hhmmss"])
            end_mmss = to_mmss(event["end_hhmmss"])

            reason_details[reason].append({
                "time_range": f"{start_mmss} - {end_mmss}",
                "start_time": start_mmss,
                "end_time": end_mmss,
                "duration_sec": round(event.get("duration_sec", 0), 2)
            })

        return {
            "success": True,
            "cheating_detected": len(events) > 0,
            "total_cheating_events": len(events),
            "cheating_events": events,
            "reason_counts": reason_counts,
            "reason_details": reason_details,
            "result_file": json_path
        }

    except Exception as e:
        print(f"Cheating detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Could not load ASR models: {str(e)}")

@app.get("/")
async def root():
    """Endpoint to check API status"""
    return {
        "message": "Video Upload API with ASR, Cheating Detection & Assessment",
        "version": "5.0.0",
        "models_loaded": models_loaded,
        "endpoints": {
            "/upload": "POST - Upload video, process ASR and cheating detection",
            "/assess": "POST - Assess answer using Gemini AI",
            "/videos": "GET - List all uploaded videos",
            "/health": "GET - Health check"
        }
    }

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), language: str = Form("en")):
    """Upload video, extract audio, process ASR and cheating detection"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if not is_allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename).stem
        file_extension = Path(file.filename).suffix
        new_filename = f"{original_filename}_{timestamp}{file_extension}"
        audio_filename = f"{original_filename}_{timestamp}.wav"
        asr_result_filename = f"{original_filename}_{timestamp}_asr.json"

        video_path = os.path.join(UPLOAD_FOLDER, new_filename)
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        asr_result_path = os.path.join(ASR_RESULT_FOLDER, asr_result_filename)

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        video_size = os.path.getsize(video_path)

        audio_extracted = extract_audio(video_path, audio_path)
        if not audio_extracted:
            raise HTTPException(status_code=500, detail="Failed to extract audio")

        audio_size = os.path.getsize(audio_path)

        if not models_loaded:
            raise HTTPException(status_code=500, detail="ASR models not loaded")

        asr_result = process_asr(audio_path, language=language)
        
        if not asr_result["success"]:
            raise HTTPException(status_code=500, detail=f"ASR failed: {asr_result.get('error')}")

        with open(asr_result_path, 'w') as f:
            json.dump(asr_result, f, indent=4)

        cheat_result = process_cheating_detection(video_path)
        if not cheat_result["success"]:
            print(f"Warning: Cheating detection failed: {cheat_result.get('error')}")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Video processed successfully",
                "data": {
                    "video": {
                        "original_filename": file.filename,
                        "saved_filename": new_filename,
                        "file_path": video_path,
                        "file_size_mb": round(video_size / (1024 * 1024), 2)
                    },
                    "audio": {
                        "filename": audio_filename,
                        "file_path": audio_path,
                        "file_size_mb": round(audio_size / (1024 * 1024), 2)
                    },
                    "asr": {
                        "full_scripts": asr_result["full_scripts"],
                        "total_speakers": asr_result["total_speakers"],
                        "speaker_list": asr_result["speaker_list"],
                        "total_duration_seconds": asr_result["total_duration_seconds"],
                        "result_file": asr_result_path
                    },
                    "cheating": {
                        "detected": cheat_result.get("cheating_detected", False),
                        "total_events": cheat_result.get("total_cheating_events", 0),
                        "events": cheat_result.get("cheating_events", []),
                        "reason_counts": cheat_result.get("reason_counts", {}),
                        "reason_details": cheat_result.get("reason_details", {}),
                        "result_file": cheat_result.get("result_file", None)
                    },
                    "upload_time": datetime.now().isoformat()
                }
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await file.close()

@app.post("/assess")
async def assess_transcript(question_id: str = Form(...), transcript: str = Form(...)):
    """Assess transcript using Gemini AI"""
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

        result = assess_answer(question_id, transcript)

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Assessment failed"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"assessment_{question_id}_{timestamp}.json"
        result_path = os.path.join(REASONING_RESULT_FOLDER, result_filename)

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "score": result["score"],
                    "reason": result["reason"],
                    "matched_level": result["matched_level"],
                    "result_file": result_path
                }
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/videos")
async def list_videos():
    """List all uploaded videos"""
    try:
        videos = []

        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)

                if os.path.isfile(file_path) and is_allowed_file(filename):
                    file_stat = os.stat(file_path)

                    videos.append({
                        "filename": filename,
                        "file_path": file_path,
                        "file_size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "created_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                    })

        return {
            "success": True,
            "total_videos": len(videos),
            "videos": videos
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ffmpeg_available = False
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        ffmpeg_available = result.returncode == 0
    except:
        pass

    config_available = os.path.exists(CONFIG_PATH)
    gemini_configured = GEMINI_API_KEY is not None

    return {
        "status": "healthy",
        "upload_folder": UPLOAD_FOLDER,
        "audio_folder": AUDIO_FOLDER,
        "asr_result_folder": ASR_RESULT_FOLDER,
        "cheat_result_folder": CHEAT_RESULT_FOLDER,
        "reasoning_result_folder": REASONING_RESULT_FOLDER,
        "models_loaded": models_loaded,
        "ffmpeg_available": ffmpeg_available,
        "config_available": config_available,
        "gemini_configured": gemini_configured
    }

if __name__ == "__main__":
    print("Starting Video Upload API with ASR, Cheating Detection & Assessment...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
