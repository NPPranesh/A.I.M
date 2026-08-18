import asyncio
import json
import websockets

# The address of your running FastAPI server
SERVER_URI = "ws://127.0.0.1:8000/ws/interview/test-session-001"

async def run_simulation():
    print(f"Connecting to Project A.I.M. Core at {SERVER_URI}...")
    
    # Open the WebSocket connection
    async with websockets.connect(SERVER_URI) as websocket:
        print("Connected successfully! Starting telemetry streams...\n")

        # Background Task 1: Stream Vision data every 1 second
        async def send_vision_data():
            while True:
                payload = {
                    "type": "VISION_TELEMETRY",
                    "payload": {"eye_contact_score": 88, "is_face_centered": True}
                }
                await websocket.send(json.dumps(payload))
                await asyncio.sleep(1.0)

        # Background Task 2: Listen for AI questions coming back from the server
        async def listen_for_server():
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                if data.get("type") == "FOLLOW_UP_QUESTION":
                    print(f"\n🔥 [A.I.M. AGENT REPLIES]: {data['payload']['text']}\n")

        # Start the background tasks
        vision_task = asyncio.create_task(send_vision_data())
        listen_task = asyncio.create_task(listen_for_server())

        # Main Task: Simulate the candidate speaking in chunks
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
            
            # Pause for 1 second between chunks to simulate natural speaking
            # (Notice this is LESS than your 2.5s threshold, so the server shouldn't interrupt!)
            await asyncio.sleep(1.0)

        print("--- Candidate goes silent (Waiting for server debouncer...) ---")
        
        # Stay connected for 5 seconds to let the server detect silence and respond
        await asyncio.sleep(5.0)

        # Clean up
        vision_task.cancel()
        listen_task.cancel()
        print("Simulation complete. Disconnecting.")

if __name__ == "__main__":
    asyncio.run(run_simulation())