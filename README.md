# Project A.I.M. 🎯
**An Edge-Computing AI Interview Mentor**

Project A.I.M. is a privacy-first, edge-computing AI interview platform. It utilizes local Large Language Models (Llama 3.2) via Ollama and a high-performance FastAPI backend to conduct real-time, STAR-method technical interviews. By processing live audio chunks and vision telemetry directly on the edge, the architecture achieves a sub-100ms latency target without relying on centralized cloud processing.

🌐 **Live Streamlit Prototype:** [aim-interview-mentor.streamlit.app](https://aim-interview-mentor.streamlit.app/)

---

## 🚀 Key Architecture Features
* **Non-Blocking WebSockets:** Bidirectional telemetry streaming between the Android client and the FastAPI backend.
* **Smart Debouncing:** A 2.5-second silence threshold timer ensures the AI only processes responses when the user naturally finishes speaking.
* **Dual-Memory Buffers:** Protects device RAM and the LLM context window by capping states (100 vision frames, 50 audio chunks).
* **Contextual RAG:** Parses candidate resumes (PDFs) to generate personalized, hyper-relevant technical questions.

---

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn, HTTPX, WebSockets
* **AI/Inference:** Ollama (Llama 3.2)
* **Mobile Client:** Android (Kotlin, Gson/Moshi)
* **Web Prototyping:** Streamlit

---

## 📡 WebSocket API Contract
**Endpoint:** `ws://<server-ip>:8001/ws/interview/{session_id}`

### 1. Vision Telemetry (Client -> Server)
Streamed continuously from the Android camera (~30fps).
```json
{
  "type": "VISION_TELEMETRY",
  "payload": {
    "eye_contact_score": 85,
    "is_face_centered": true
  }
}
```

### 2. Audio Telemetry (Client -> Server)
Streamed in real-time as the user speaks.
```json
{
  "type": "AUDIO_TELEMETRY",
  "payload": {
    "transcript_chunk": "I deployed the backend using Docker."
  }
}
```

### 3. AI Response (Server -> Client)
Triggered automatically after 2.5 seconds of audio silence.
```json
{
  "type": "FOLLOW_UP_QUESTION",
  "payload": {
    "text": "Maintain steady eye contact. How did you handle the Docker network mapping?"
  }
}
```

---

## 💻 Local Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Local Ollama:**
   Ensure Ollama is installed and running Llama 3.2 in the background.
   ```bash
   ollama run llama3.2
   ```
3. **Run the FastAPI Server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```

---

## 👥 The Team
Built by B.Tech Information Technology students at VIT Vellore.
* **Pranesh NP:** Technical Lead & Backend Infrastructure
* **Viswajith:** AI Engineering & RAG Implementation
* **Vaisanth Ragav:** Android Development (Audio/Speech Pipeline)
* **Jaiarvindhan:** Android Development (Vision Pipeline)
* **Dhevananth:** UI/UX Design & Web Prototyping
