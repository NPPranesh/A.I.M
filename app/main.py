from pathlib import Path
import sqlite3
import hashlib
import secrets
import hmac
import json
import sys
import asyncio
import httpx
import logging
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing core modules
from app.ai_agent import load_resume_text, generate_follow_up

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from aim_ai import AIMAIError, evaluate_star_answer, generate_interview_question, ingest_resume

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "private_uploads"
DB_PATH = DATA_DIR / "aim.sqlite3"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AIM_Core")

http_client = None

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

# --- DB Dependency for REST Endpoints ---
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()

# --- Pydantic Schemas for Android REST Client ---
class UserAuth(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class InterviewStartRequest(BaseModel):
    user_id: int
    interview_type: str
    difficulty: str = "Adaptive"
    question_count: int = 10
    immediate_eval: bool = True

class QuestionGenerateRequest(BaseModel):
    user_id: int
    interview_type: str
    difficulty: str = "Adaptive"
    previous_answers: List[dict] = []

class EvaluationRequest(BaseModel):
    user_id: int
    question: str
    answer: str

def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000).hex()

# ==========================================
# 1. NEW REST ENDPOINTS FOR ANDROID CLIENT
# ==========================================

@app.post("/api/auth/register")
def register(user: UserAuth, db: sqlite3.Connection = Depends(get_db)):
    if not user.name:
        raise HTTPException(status_code=400, detail="Name is required.")
    salt = secrets.token_bytes(16)
    try:
        cursor = db.execute(
            "INSERT INTO users(name, email, password_hash, salt) VALUES(?,?,?,?)",
            (user.name.strip(), user.email.strip().lower(), hash_password(user.password, salt), salt.hex())
        )
        db.commit()
        return {"status": "success", "user_id": cursor.lastrowid, "name": user.name}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already exists.")

@app.post("/api/auth/login")
def login(user: UserAuth, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM users WHERE email=?", (user.email.strip().lower(),)).fetchone()
    if not row or not hmac.compare_digest(hash_password(user.password, bytes.fromhex(row["salt"])), row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"status": "success", "user_id": row["id"], "name": row["name"], "email": row["email"]}

@app.post("/api/interview/start")
def start_interview(req: InterviewStartRequest, db: sqlite3.Connection = Depends(get_db)):
    db.execute("DELETE FROM interview_drafts WHERE user_id=?", (req.user_id,))
    cursor = db.execute(
        "INSERT INTO interview_drafts(user_id, interview_type, difficulty, duration, immediate_eval) VALUES(?,?,?,?,?)",
        (req.user_id, req.interview_type, req.difficulty, str(req.question_count), 1 if req.immediate_eval else 0)
    )
    db.commit()
    return {"status": "success", "draft_id": cursor.lastrowid, "total_questions": req.question_count}

@app.post("/api/interview/question")
def get_next_question(req: QuestionGenerateRequest, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT stored_path FROM resumes WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (req.user_id,)).fetchone()
    if not row or not Path(row["stored_path"]).exists():
        raise HTTPException(status_code=400, detail="Upload a resume first.")
    
    try:
        resume_ctx = ingest_resume(Path(row["stored_path"]))
        question_data = generate_interview_question(
            resume_context=resume_ctx,
            interview_type=req.interview_type,
            difficulty=req.difficulty,
            previous_answers=req.previous_answers
        )
        return {"status": "success", "data": question_data}
    except AIMAIError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interview/evaluate")
def evaluate_answer(req: EvaluationRequest, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT stored_path FROM resumes WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (req.user_id,)).fetchone()
    resume_ctx = ingest_resume(Path(row["stored_path"])) if row and Path(row["stored_path"]).exists() else None
    
    try:
        feedback_data = evaluate_star_answer(
            question=req.question,
            answer=req.answer,
            resume_context=resume_ctx
        )
        return {"status": "success", "feedback": feedback_data}
    except AIMAIError as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 2. EXISTING WEBSOCKET INTERVIEW ORCHESTRATOR
# ==========================================

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
    resume_path = None
    resolved_path = resume_path or f"resumes/{session_id}.pdf"
    state.resume_context = load_resume_text(resolved_path)
    print(f"[A.I.M. Brain] Resume context loaded for {session_id} "
          f"({len(state.resume_context)} chars) from '{resolved_path}'")

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