import os
# Suppress Hugging Face Windows Symlink & Unauthenticated Logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


import numpy as np
import asyncio
import platform
import httpx
import logging
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import ctranslate2
from faster_whisper import WhisperModel
from app.ai_agent import load_resume_text, generate_follow_up

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AIM_Core")


# Detect GPU Hardware Acceleration via CTranslate2
if ctranslate2.get_cuda_device_count() > 0:
    device = "cuda"
    compute_type = "float16"
    logger.info("Running on Windows/Linux NVIDIA CUDA (FP16)")
elif platform.system() == "Darwin":  # macOS
    device = "cpu"
    compute_type = "int8"  # int8 gives native speedups on Apple Silicon
    logger.info("Running on macOS (Apple Silicon CPU/int8)")
else:
    device = "cpu"
    compute_type = "int8"
    logger.info("Running on generic CPU fallback")

# Upgraded to 'medium' for vastly superior Indian English & proper noun accuracy
stt_model = WhisperModel("medium", device=device, compute_type=compute_type)

http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient()
    yield
    await http_client.aclose()

app = FastAPI(title="Project A.I.M. Orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/telemetry/process-chunk")
async def process_telemetry_chunk(file: UploadFile = File(...)):
    pcm_bytes = await file.read()
    
    # 1. Byte length guard for 16-bit PCM alignment
    if not pcm_bytes or len(pcm_bytes) % 2 != 0:
        return {
            "status": "success", 
            "telemetry": {
                "rms_amplitude": 0.0, 
                "zero_crossing_rate": 0.0, 
                "spectral_centroid_hz": 0.0, 
                "transcript": ""
            }
        }

    # 2. Decode int16 Little-Endian PCM stream safely
    audio_int16 = np.frombuffer(pcm_bytes, dtype='<i2')
    
    if len(audio_int16) < 1600:  # Minimum length check (~0.1s)
        return {
            "status": "success", 
            "telemetry": {
                "rms_amplitude": 0.0, 
                "zero_crossing_rate": 0.0, 
                "spectral_centroid_hz": 0.0, 
                "transcript": ""
            }
        }

    # Normalize to float32 (-1.0 to 1.0)
    audio_data = audio_int16.astype(np.float32) / 32768.0

    # 3. RMS Calculation
    rms = float(np.sqrt(np.mean(np.square(audio_data))))

    # 4. Zero Crossing Rate (ZCR) with zero-mean centering
    centered_audio = audio_data - np.mean(audio_data)
    zero_crossings = np.diff(np.signbit(centered_audio))
    zcr = float(np.sum(zero_crossings) / len(audio_data))

    # 5. Spectral Centroid Calculation (FFT)
    sample_rate = 16000
    fft_spectrum = np.abs(np.fft.rfft(audio_data))
    freq_bins = np.fft.rfftfreq(len(audio_data), d=1.0 / sample_rate)

    fft_sum = np.sum(fft_spectrum)
    spectral_centroid = float(np.sum(freq_bins * fft_spectrum) / fft_sum) if fft_sum > 1e-6 else 0.0

    # 6. Whisper GPU Transcription optimized for Indian English
    transcript = ""
    if rms > 0.015:
        segments, _ = stt_model.transcribe(
            audio_data,
            beam_size=5,  # Higher beam size improves accuracy for accented speech
            language="en",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=250,
                threshold=0.4
            ),
            # Priming Whisper with domain context & Indian English vocabulary hints
            initial_prompt="A software engineering technical interview conversation in Indian English involving Sabari and team members.",
            
            # Anti-Hallucination & Repetition Control
            condition_on_previous_text=False,
            no_speech_threshold=0.35,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.4
        )
        transcript = " ".join([segment.text for segment in segments]).strip()

    if transcript:
        logger.info(f" [Whisper Live]: {transcript}")

    return {
        "status": "success",
        "telemetry": {
            "rms_amplitude": round(rms, 4),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_centroid_hz": round(spectral_centroid, 1),
            "transcript": transcript
        }
    }

# --- EXISTING WEBSOCKET LOGIC ---
class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_transcript = [] 
        self.eye_contact_history = []
        self.resume_context: str = "(No resume was provided for this candidate.)"
        self.silence_threshold_seconds = 2.5 
        self.debounce_task: Optional[asyncio.Task] = None

class ConnectionManager:
    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}
        self.states: Dict[str, SessionState] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_sessions[session_id] = websocket
        self.states[session_id] = SessionState(session_id)
        logger.info(f"Client {session_id} connected to A.I.M.")

    def disconnect(self, session_id: str):
        state = self.states.get(session_id)
        if state and state.debounce_task:
            state.debounce_task.cancel()
        self.active_sessions.pop(session_id, None)
        self.states.pop(session_id, None)
        logger.info(f"Client {session_id} disconnected.")

    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_sessions:
            await self.active_sessions[session_id].send_json(data)

manager = ConnectionManager()

async def evaluate_and_respond(session_id: str):
    state = manager.states.get(session_id)
    if not state or not state.current_transcript:
        return

    candidate_answer = " ".join(state.current_transcript)
    avg_eye_contact = sum(state.eye_contact_history) // len(state.eye_contact_history) if state.eye_contact_history else 100

    logger.info(f"Processing Answer: '{candidate_answer}' (Avg Eye Contact: {avg_eye_contact}%)")

    try:
        final_response = await generate_follow_up(
            candidate_answer=candidate_answer,
            eye_contact=avg_eye_contact,
            resume_context=state.resume_context
        )

        await manager.send_json(session_id, {
            "type": "FOLLOW_UP_QUESTION", 
            "payload": {"text": final_response}
        })
    except Exception as e:
        logger.error(f"AI Agent Inference Failed: {e}")

    state.current_transcript.clear()
    state.eye_contact_history.clear()
    logger.info("Memory wiped. Listening for next answer...")

async def trigger_silence_countdown(session_id: str, delay_seconds: float):
    try:
        await asyncio.sleep(delay_seconds)
        await evaluate_and_respond(session_id)
    except asyncio.CancelledError:
        pass

@app.websocket("/ws/interview/{session_id}")
async def interview_endpoint(websocket: WebSocket, session_id: str = "default_session"):
    await manager.connect(session_id, websocket)
    state = manager.states[session_id]
    
    resolved_path = f"resumes/{session_id}.pdf"
    try:
        state.resume_context = await asyncio.to_thread(load_resume_text, resolved_path)
    except Exception as e:
        logger.warning(f"Could not load resume at {resolved_path}: {e}")

    logger.info(f"[A.I.M. Brain] Resume loaded for {session_id} ({len(state.resume_context)} chars)")

    try:
        while True:
            client_telemetry = await websocket.receive_json()
            
            chunk = client_telemetry.get("payload", {}).get("transcript_chunk", "")
            eye_score = client_telemetry.get("payload", {}).get("eye_contact_score", 100)

            state.eye_contact_history.append(eye_score)
            if len(state.eye_contact_history) > 100:
                state.eye_contact_history.pop(0)

            if chunk.strip():
                state.current_transcript.append(chunk)
                if len(state.current_transcript) > 50:
                    state.current_transcript.pop(0)
                
                if state.debounce_task:
                    state.debounce_task.cancel()
                
                state.debounce_task = asyncio.create_task(
                    trigger_silence_countdown(session_id, state.silence_threshold_seconds)
                )
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)