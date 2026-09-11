from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx
from typing import Dict, Optional
from contextlib import asynccontextmanager
import logging

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

RESUME_CONTEXT = "Candidate is a B.Tech IT student at VIT Vellore experienced in Python, FastAPI, and C programming."

class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_transcript = [] 
        self.eye_contact_history = []
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
    system_prompt = (
        "You are an expert technical interviewer conducting a mock interview for Project A.I.M. "
        "Strictly adhere to these operational constraints:\n"
        f"1. CANDIDATE RESUME CONTEXT:\n{RESUME_CONTEXT}\n"
        "2. Internally evaluate the candidate's last answer using the STAR method, but DO NOT output the STAR breakdown.\n"
        "3. Output ONLY a single, direct, challenging technical follow-up question grounded in their resume and latest answer.\n"
        "4. Keep your total response under 25 words. Never explain your reasoning."
    )
    
    tech_payload = {
        "model": "llama3.2",
        "prompt": f"{system_prompt}\n\nCandidate Answer: {candidate_answer}\nInterviewer:",
        "stream": False
    }
    
    tech_task = http_client.post("http://localhost:11434/api/generate", json=tech_payload, timeout=60.0)
    tasks = [tech_task]

    if avg_eye_contact < 70:
        behavioral_prompt = (
            f"The candidate's eye contact dropped to {avg_eye_contact}%. "
            "Write a single, encouraging sentence reminding them to look at the camera. "
            "Do not apologize or explain."
        )
        beh_payload = {
            "model": "llama3.2",
            "prompt": behavioral_prompt,
            "stream": False
        }
        
        beh_task = http_client.post("http://localhost:11434/api/generate", json=beh_payload, timeout=10.0)
        tasks.append(beh_task)

    try:
        # Restore the parsing and sending logic!
        results = await asyncio.gather(*tasks)
        
        technical_question = results[0].json().get("response", "").strip()
        
        if len(results) > 1:
            dynamic_warning = results[1].json().get("response", "").strip()
            final_response = f"{dynamic_warning} {technical_question}"
        else:
            final_response = technical_question

        # Send back to Android
        await manager.send_json(session_id, {
            "type": "FOLLOW_UP_QUESTION", 
            "payload": {"text": final_response}
        })
    except Exception as e:
       logger.error(f"Ollama Inference Failed: {e}")

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