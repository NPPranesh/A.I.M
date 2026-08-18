from app.ai_agent import generate_follow_up
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
import time
from typing import Dict, Optional

# 1. Initialize the Project A.I.M. Server
app = FastAPI(title="Project A.I.M. Orchestrator")

# 2. Define the Memory State
class SessionState:
    """Holds the live data for a single interview session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Buffers for incoming Android telemetry
        self.current_transcript = [] 
        self.eye_contact_history = []
        
        # Turn-taking configuration
        self.silence_threshold_seconds = 2.5 
        self.debounce_task: Optional[asyncio.Task] = None

class ConnectionManager:
    """Tracks active phones connected to the A.I.M. backend."""
    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}
        self.states: Dict[str, SessionState] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_sessions[session_id] = websocket
        self.states[session_id] = SessionState(session_id)
        print(f"[+] Client {session_id} connected to A.I.M.")

    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.states:
            del self.states[session_id]
        print(f"[-] Client {session_id} disconnected.")

    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_sessions:
            await self.active_sessions[session_id].send_json(data)

# Create a single global manager
manager = ConnectionManager()

async def evaluate_and_respond(session_id: str):
    state = manager.states.get(session_id)
    if not state or not state.current_transcript:
        return

    full_answer = " ".join(state.current_transcript)
    
    # 1. Calculate the average eye contact!
    if len(state.eye_contact_history) > 0:
        avg_eye_contact = sum(state.eye_contact_history) // len(state.eye_contact_history)
    else:
        avg_eye_contact = 100 # Default if no data arrived

    print(f"\n[A.I.M. Brain] User answered: '{full_answer}' (Eye Contact: {avg_eye_contact}%)")

    try:
        # 2. Pass BOTH the text and the vision score to Member 5's AI
        ai_question = await generate_follow_up(full_answer, avg_eye_contact)
        
        await manager.send_json(session_id, {
            "type": "FOLLOW_UP_QUESTION",
            "payload": {"text": ai_question}
        })
        
    except Exception as e:
        print(f"[Error] Gemini API failed: {e}")
    
    # 3. Wipe BOTH memory buckets clean for the next question
    state.current_transcript.clear()
    state.eye_contact_history.clear()
    print("[A.I.M. Brain] Memory wiped. Listening for next answer...\n")

async def trigger_silence_countdown(session_id: str, delay_seconds: float):
    """The background timer that gets reset every time the user speaks."""
    try:
        await asyncio.sleep(delay_seconds)
        # If we survive the sleep without being cancelled, trigger the AI!
        await evaluate_and_respond(session_id)
    except asyncio.CancelledError:
        # The user spoke again! This timer gets cancelled and dies silently.
        pass

@app.websocket("/ws/interview/{session_id}")
async def interview_endpoint(websocket: WebSocket, session_id: str):
    # Open the door and create the memory bucket
    await manager.connect(session_id, websocket)
    state = manager.states[session_id]

    try:
        while True:
            # Wait for data to arrive from the phone
            raw_data = await websocket.receive_text()
            message = json.loads(raw_data)
            
            msg_type = message.get("type")
            payload = message.get("payload", {})

            # ROUTE 1: Vision Data
            if msg_type == "VISION_TELEMETRY":
                score = payload.get("eye_contact_score", 100)
                state.eye_contact_history.append(score)
                # We don't trigger AI here, just collect the stats.

            # ROUTE 2: Audio Data
            elif msg_type == "AUDIO_TELEMETRY":
                chunk = payload.get("transcript_chunk", "")
                if chunk.strip():
                    state.current_transcript.append(chunk)
                    print(f"[Audio] Received chunk: {chunk}")

                    # THE DEBOUNCER LOGIC:
                    # 1. Kill the old timer because the user is still speaking
                    if state.debounce_task and not state.debounce_task.done():
                        state.debounce_task.cancel()

                    # 2. Start a fresh countdown timer
                    state.debounce_task = asyncio.create_task(
                        trigger_silence_countdown(session_id, state.silence_threshold_seconds)
                    )

    except WebSocketDisconnect:
        # If the app crashes or drops signal, clean up the memory
        manager.disconnect(session_id)