import asyncio
import json
import websockets
import logging

# Configure client-side logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | CLIENT | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AIM_Client")

SERVER_URI = "ws://127.0.0.1:8001/ws/interview/test-session-001"

async def run_simulation():
    logger.info(f"Connecting to Project A.I.M. Core at {SERVER_URI}...")
    
    async with websockets.connect(SERVER_URI, origin=None) as websocket:
        logger.info("Connected successfully! Starting telemetry streams...")

        async def send_vision_data():
            try:
                while True:
                    payload = {
                        "type": "VISION_TELEMETRY",
                        "payload": {"eye_contact_score": 23, "is_face_centered": True}
                    }
                    await websocket.send(json.dumps(payload))
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                pass 

        async def listen_for_server():
            try:
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    if data.get("type") == "FOLLOW_UP_QUESTION":
                        logger.info(f"🔥 [A.I.M. AGENT REPLIES]: {data['payload']['text']}")
            except asyncio.CancelledError:
                pass

        vision_task = asyncio.create_task(send_vision_data())
        listen_task = asyncio.create_task(listen_for_server())

        speech_chunks = [
            "I used Docker to containerize",
            " our backend services",
            " and deployed them to AWS."
        ]

        logger.info("--- Candidate starts speaking ---")
        for chunk in speech_chunks:
            payload = {
                "type": "AUDIO_TELEMETRY",
                "payload": {"transcript_chunk": chunk}
            }
            await websocket.send(json.dumps(payload))
            logger.info(f"Sent audio chunk: '{chunk}'")
            await asyncio.sleep(1.0)

        logger.info("--- Candidate goes silent (Waiting for server debouncer...) ---")
        
        vision_task.cancel()
        
        await asyncio.sleep(5.0)

        listen_task.cancel()
        logger.info("Simulation complete. Disconnecting.")

if __name__ == "__main__":
    asyncio.run(run_simulation())