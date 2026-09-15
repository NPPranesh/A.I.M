import requests

BASE_URL = "http://127.0.0.1:8000/api"

def test_backend_flow():
    # 1. Login Test
    auth_resp = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "pranesh.np2025@vitstudent.ac.in",
        "password": "12345678"
    })
    print("Login Response:", auth_resp.json())

    if auth_resp.status_code == 200:
        user_id = auth_resp.json()["user_id"]
        
        # 2. Start Interview Draft Test
        start_resp = requests.post(f"{BASE_URL}/interview/start", json={
            "user_id": user_id,
            "interview_type": "Technical",
            "difficulty": "Adaptive",
            "question_count": 5
        })
        print("Start Interview Response:", start_resp.json())

if __name__ == "__main__":
    test_backend_flow()