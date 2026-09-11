import asyncio
import json
import websockets

SERVER_URI = "ws://127.0.0.1:8001/ws/interview/test-session-001"

async def run_simulation():
    print(f"Connecting to Project A.I.M. Core at {SERVER_URI}...")
    
    async with websockets.connect(SERVER_URI, origin=None) as websocket:
        print("Connected successfully! Starting telemetry streams...\n")

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
                        print(f"\n🔥 [A.I.M. AGENT REPLIES]: {data['payload']['text']}\n")
            except asyncio.CancelledError:
                pass

        vision_task = asyncio.create_task(send_vision_data())
        listen_task = asyncio.create_task(listen_for_server())

        speech_chunks = [
            "I used Docker to containerize",
            " our backend services",
            " and deployed them to AWS."
        ]

        print("--- Candidate starts speaking ---")
        for chunk in speech_chunks:
            payload = {
                "type": "AUDIO_TELEMETRY",
                "payload": {"transcript_chunk": chunk}
            }
            await websocket.send(json.dumps(payload))
            print(f"Sent audio chunk: '{chunk}'")
            await asyncio.sleep(1.0)

        print("--- Candidate goes silent (Waiting for server debouncer...) ---")
        
        vision_task.cancel()
    
        await asyncio.sleep(5.0)

        listen_task.cancel()
        print("Simulation complete. Disconnecting.")

if __name__ == "__main__":
    asyncio.run(run_simulation())