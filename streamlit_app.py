from __future__ import annotations

from datetime import date
import hashlib
import subprocess
import sys
import hmac
import json
from pathlib import Path
import secrets
import sqlite3

import streamlit as st
from aim_ai import AIMAIError, evaluate_star_answer, generate_interview_question, ingest_resume

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "private_uploads"
DB_PATH = DATA_DIR / "aim.sqlite3"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TOTAL_QUESTIONS = 10

st.set_page_config(
    page_title="AIM | Adaptive Interview Mentor",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--navy:#0b1020;--surface:#11182b;--surface2:#17213a;--line:rgba(157,176,218,.16);--text:#f3f6ff;--muted:#8f9ab4;--indigo:#7168ff;--cyan:#36d9e8;--green:#67d8a1;--amber:#f3b766}
.stApp{background:radial-gradient(circle at 78% -15%,rgba(79,72,195,.18),transparent 35%),var(--navy);color:var(--text)}
[data-testid="stSidebar"]{background:rgba(10,15,32,.9);border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em}.stMarkdown,.stTextInput,.stSelectbox,.stTextArea,.stFileUploader,.stButton{font-family:'DM Sans',sans-serif}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(23,33,58,.9),rgba(17,24,43,.72));border:1px solid var(--line);border-radius:10px;padding:16px}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif}.stButton>button{border-radius:7px;border:1px solid var(--line);min-height:42px;font-weight:600}.stButton>button[kind="primary"]{background:var(--indigo);border-color:var(--indigo);color:white}.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]{background:rgba(255,255,255,.035);border-color:var(--line);color:var(--text)}
.aim-card{background:linear-gradient(145deg,rgba(23,33,58,.88),rgba(17,24,43,.75));border:1px solid var(--line);border-radius:11px;padding:22px;height:100%}.eyebrow{color:var(--cyan);font-size:.7rem;letter-spacing:.13em;font-weight:700}.muted{color:var(--muted)}.score{font:600 56px 'Space Grotesk',sans-serif;letter-spacing:-.06em}.pill{display:inline-block;padding:5px 9px;border-radius:5px;background:rgba(103,216,161,.1);border:1px solid rgba(103,216,161,.22);color:var(--green);font-size:.75rem}.auth-wrap{max-width:430px;margin:8vh auto}.auth-card{background:linear-gradient(145deg,rgba(23,33,58,.95),rgba(17,24,43,.86));border:1px solid var(--line);border-radius:14px;padding:32px}.brand{font:700 24px 'Space Grotesk',sans-serif;letter-spacing:.06em}.brand-mark{display:inline-grid;place-items:center;width:31px;height:31px;margin-right:9px;background:linear-gradient(145deg,#8d86ff,#4540b9);clip-path:polygon(50% 0,100% 100%,72% 100%,50% 36%,28% 100%,0 100%)}.stProgress>div>div>div{background:linear-gradient(90deg,var(--indigo),var(--cyan))}
.skill-tag{display:inline-block;padding:6px 12px;margin:4px 6px 4px 0;background:rgba(113,104,255,0.15);border:1px solid rgba(113,104,255,0.35);border-radius:16px;color:#f3f6ff;font-size:0.85rem}
</style>
""", unsafe_allow_html=True)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    with get_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, salt TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS resumes (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT NOT NULL, stored_path TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id))")
        conn.execute("CREATE TABLE IF NOT EXISTS interviews (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, interview_type TEXT NOT NULL, score INTEGER NOT NULL, duration TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id))")
        conn.execute("CREATE TABLE IF NOT EXISTS interview_drafts (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL UNIQUE, interview_type TEXT NOT NULL, difficulty TEXT NOT NULL, duration TEXT NOT NULL, immediate_eval INTEGER NOT NULL DEFAULT 1, question_index INTEGER NOT NULL DEFAULT 0, answers_json TEXT NOT NULL DEFAULT '[]', updated_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id))")
        
        # Safe migration for pre-existing local SQLite files
        try:
            conn.execute("ALTER TABLE interview_drafts ADD COLUMN immediate_eval INTEGER NOT NULL DEFAULT 1;")
        except sqlite3.OperationalError:
            pass

def password_hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000).hex()


def create_user(name: str, email: str, password: str) -> bool:
    salt = secrets.token_bytes(16)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users(name,email,password_hash,salt) VALUES(?,?,?,?)",
                (name.strip(), email.strip().lower(), password_hash(password, salt), salt.hex()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate(email: str, password: str) -> dict | None:
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if not row or not hmac.compare_digest(password_hash(password, bytes.fromhex(row["salt"])), row["password_hash"]):
        return None
    return dict(row)


def current_user() -> dict | None:
    return st.session_state.get("user")


def user_interviews(user_id: int) -> list[dict]:
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT interview_type, score, duration, created_at FROM interviews WHERE user_id=? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_user_resume_path(user_id: int) -> Path | None:
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT stored_path FROM resumes WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    if row and Path(row["stored_path"]).exists():
        return Path(row["stored_path"])
    return None


def get_or_load_resume_context(user_id: int):
    if st.session_state.get("resume_context"):
        return st.session_state["resume_context"]
    resume_path = get_user_resume_path(user_id)
    if resume_path:
        try:
            ctx = ingest_resume(resume_path)
            st.session_state.resume_context = ctx
            return ctx
        except AIMAIError:
            return None
    return None


def active_draft(user_id: int) -> dict | None:
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM interview_drafts WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return None
    draft = dict(row)
    draft["answers"] = json.loads(draft.pop("answers_json"))
    draft["type"] = draft.get("interview_type")
    draft["immediate_eval"] = bool(draft.get("immediate_eval", 1))
    
    try:
        draft["total_questions"] = int(draft.get("duration", 10))
    except (ValueError, TypeError):
        draft["total_questions"] = 10
    return draft

def start_draft(user_id: int, interview: dict) -> dict:
    total_q = int(interview.get("question_count", 10))
    with get_db() as conn:
        conn.execute("DELETE FROM interview_drafts WHERE user_id=?", (user_id,))
        cursor = conn.execute(
            "INSERT INTO interview_drafts(user_id,interview_type,difficulty,duration,immediate_eval) VALUES(?,?,?,?,?)",
            (
                user_id, 
                interview["type"], 
                interview["difficulty"], 
                str(total_q),
                1 if interview.get("immediate_eval", True) else 0
            ),
        )
        draft_id = cursor.lastrowid
    
    return {**interview, "id": draft_id, "question_index": 0, "answers": [], "total_questions": total_q}


def save_draft(draft: dict) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE interview_drafts SET question_index=?, answers_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (draft["question_index"], json.dumps(draft["answers"]), draft["id"]),
        )


def delete_draft(user_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM interview_drafts WHERE user_id=?", (user_id,))


def readiness_data(user_id: int) -> tuple[int, list[dict], bool]:
    records = user_interviews(user_id)
    if not records:
        return 0, [], False
    average = round(sum(record["score"] for record in records) / len(records))
    return average, records, True


def navigate_to(page_name: str) -> None:
    st.session_state.current_page = page_name
    st.rerun()


def auth_view() -> None:
    st.markdown('<div class="auth-wrap"><div class="auth-card"><div class="brand"><span class="brand-mark">A</span>AIM</div><p class="muted">Adaptive Interview Mentor</p>', unsafe_allow_html=True)
    st.markdown("### Welcome to AIM")
    st.caption("Create a private practice space that gets sharper with every interview.")
    mode = st.radio("Account action", ["Create account", "Sign in"], horizontal=True, label_visibility="collapsed")
    
    with st.form("auth_form"):
        name = st.text_input("Your name", placeholder="Enter your name") if mode == "Create account" else ""
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        
        submitted = st.form_submit_button(
            "Create my AIM account" if mode == "Create account" else "Sign in",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not email or len(password) < 8 or (mode == "Create account" and not name):
            st.error("Enter all required fields. Passwords must be at least 8 characters.")
        elif mode == "Create account":
            if create_user(name, email, password):
                st.session_state.user = authenticate(email, password)
                st.session_state.current_page = "Home"
                st.rerun()
            st.error("An account with that email already exists. Sign in instead.")
        else:
            user = authenticate(email, password)
            if user:
                st.session_state.user = user
                st.session_state.current_page = "Home"
                st.rerun()
            st.error("Email or password is incorrect.")
            
    st.caption("Passwords are hashed server-side with PBKDF2 and are never stored in plain text.")
    st.markdown("</div></div>", unsafe_allow_html=True)


def sidebar() -> None:
    user = current_user()
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">A</span>AIM</div><p class="muted">Adaptive Interview Mentor</p>', unsafe_allow_html=True)
        st.write(f"Welcome, **{user['name']}**")
        
        pages = ["Home", "Interviews", "Progress", "Profile", "Resume", "Settings"]
        current_idx = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
        selected_page = st.radio("Navigate", pages, index=current_idx, label_visibility="collapsed")
        
        if selected_page != st.session_state.current_page and st.session_state.current_page in pages:
            navigate_to(selected_page)
            
        st.divider()
        st.caption("AIM is learning · 3 insights updated today")
        if st.button("Sign out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def home() -> None:
    readiness, records, has_history = readiness_data(current_user()["id"])
    today_str = date.today().strftime("%A, %B %d, %Y").upper()
    st.markdown(f'<p class="eyebrow">{today_str}</p>', unsafe_allow_html=True)
    st.title(f"Good morning, {current_user()['name']}.")
    st.caption("Your next interview starts here.")
    if st.button("Build an interview", type="primary"):
        navigate_to("Interview setup")
    st.write("")
    left, right = st.columns([1.15, 0.85])
    with left:
        with st.container(border=True):
            st.markdown('<p class="eyebrow">YOUR CURRENT SIGNAL</p>', unsafe_allow_html=True)
            st.subheader("Interview readiness")
            if has_history:
                st.markdown(f'<div class="score">{readiness} <span class="muted" style="font-size:16px">/100</span></div><span class="pill">{len(records)} completed interview(s)</span><p class="muted">based on your completed practice</p>', unsafe_allow_html=True)
                st.progress(readiness / 100, text="Technical confidence is rising")
            else:
                st.markdown('<div class="score">0 <span class="muted" style="font-size:16px">/100</span></div><p class="muted">Complete your first interview to build your readiness score.</p>', unsafe_allow_html=True)
                st.progress(0, text="No interview signal yet")
    with right:
        with st.container(border=True):
            st.markdown('<p class="eyebrow">PERSONALIZED FOCUS</p>', unsafe_allow_html=True)
            st.subheader("AIM noticed")
            if has_history:
                st.markdown("**AIM is finding your first focus area.**")
                st.caption("Complete more interviews and AIM will identify the skill with the most room to grow.")
                st.info("Recommended focus: complete another interview")
            else:
                st.markdown("**Your personalized focus is waiting.**")
                st.caption("AIM needs your first interview before it can identify a meaningful weakness.")
                st.info("Complete your first interview to unlock insights")
    st.subheader("Keep your momentum")
    with st.container(border=True):
        st.markdown('<p class="eyebrow">YOUR NEXT STEP</p>', unsafe_allow_html=True)
        st.subheader("Start your first adaptive interview" if not has_history else "AI/ML Technical Interview")
        st.caption("Adaptive · 20 min · AIM will learn from your answers")
        if st.button("Start interview", type="primary"):
            navigate_to("Interview setup")
    st.subheader("Skill snapshot")
    skills = {"Python": 0.0, "Machine Learning": 0.0, "SQL": 0.0, "Statistics": 0.0} if not has_history else {"Python": 0.85, "Machine Learning": 0.71, "SQL": 0.48, "Statistics": 0.52}
    for skill, value in skills.items():
        st.write(f"**{skill}** · {value:.0%}")
        st.progress(value)


def interview_setup() -> None:
    st.markdown('<p class="eyebrow">MAKE IT YOURS</p>', unsafe_allow_html=True)
    st.title("Build your interview")
    st.caption("AIM will adapt the path as you answer.")
    draft = active_draft(current_user()["id"])
    if draft:
        total_q = draft.get("total_questions", 10)
        st.info(f"You have an unfinished {draft['interview_type']} interview at question {draft['question_index'] + 1} of {total_q}.")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            if st.button("Continue interview", type="primary"):
                st.session_state.interview = draft
                navigate_to("Live interview")
        with col2:
            if st.button("Discard draft"):
                delete_draft(current_user()["id"])
                st.rerun()
        st.divider()
        
    with st.form("interview_setup"):
        interview_type = st.selectbox("Interview type", ["Technical", "Behavioral", "Project Defense", "DSA"])
        difficulty = st.radio("Difficulty", ["Beginner", "Adaptive", "Advanced"], index=1, horizontal=True)
        
        # New 4-option question count selector
        question_count = st.radio("Number of questions", ["5", "10", "15", "20"], index=1, horizontal=True)
        
        immediate_eval = st.toggle("Immediate Evaluation after each question", value=True, help="If disabled, you will receive a full batch evaluation at the end.")
        
        submitted = st.form_submit_button("Start with AIM", type="primary", use_container_width=True)
        
    if submitted:
        st.session_state.interview = start_draft(
            current_user()["id"],
            {
                "type": interview_type, 
                "difficulty": difficulty or "Adaptive", 
                "question_count": question_count or "10",
                "immediate_eval": immediate_eval
            },
        )
        navigate_to("Live interview")

def live_interview() -> None:
    interview = st.session_state.get("interview")
    if not interview:
        draft = active_draft(current_user()["id"])
        if draft:
            interview = draft
            st.session_state.interview = draft
        else:
            navigate_to("Interview setup")
            return

    question_index = interview.get("question_index", 0)
    total_questions = interview.get("total_questions", 10)

    resume_context = get_or_load_resume_context(current_user()["id"])
    if not resume_context:
        st.warning("Please upload a resume first so AIM can tailor your questions.")
        if st.button("Upload resume now", type="primary"):
            navigate_to("Resume")
        return

    interview_type = interview.get("interview_type") or interview.get("type", "Technical")
    st.markdown(f"**AIM** · {interview_type} interview · Question {question_index + 1} of {total_questions}")    
    st.progress((question_index + 1) / total_questions)
    st.markdown("<div style='text-align:center;font-size:48px;color:#36d9e8'>✦</div><p style='text-align:center;color:#8f9ab4'>AIM is listening</p>", unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">PERSONALIZED FROM YOUR PROFILE</p>', unsafe_allow_html=True)

    question_cache = st.session_state.setdefault("question_cache", {})
    cache_key = f"{interview.get('id', 'new')}:{question_index}"
    if cache_key not in question_cache:
        try:
            with st.spinner("AIM is preparing your next question…"):
                question_cache[cache_key] = generate_interview_question(
                    resume_context=resume_context,
                    interview_type=interview.get("type") or interview.get("interview_type", "Technical"),
                    difficulty=interview.get("difficulty", "Adaptive"),
                    previous_answers=interview.get("answers", []),
                )
        except AIMAIError as exc:
            st.error(str(exc))
            return

    question_data = question_cache[cache_key]

    if "vision_process" not in st.session_state or st.session_state.vision_process.poll() is not None:
        try:
            tracker_path = APP_DIR / "app" / "services" / "vision" / "eye_tracker.py"
            st.session_state.vision_process = subprocess.Popen(
                [sys.executable, str(tracker_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            st.error(f"Failed to start camera: {e}")

    # Layout container block to enforce single clean rendering
    with st.container():
        st.markdown(f"### {question_data['question']}")
        st.caption(f"Focus: {question_data['focus']}")
        
        with st.form(key=f"unique_answer_form_{question_index}"):
            user_response = st.text_area(
                "Your answer", 
                placeholder="Be specific. AIM will follow your thinking.", 
                height=150,
                label_visibility="collapsed"
            )
            col1, col2 = st.columns(2)
            with col1:
                skip = st.form_submit_button("Skip", use_container_width=True)
            with col2:
                submit = st.form_submit_button("Submit answer", type="primary", use_container_width=True)
                
        if submit or skip:
            submitted_answer = user_response.strip() if submit and user_response.strip() else "Skipped"
            st.session_state.current_question = question_data["question"]
            st.session_state.current_answer = submitted_answer

            is_immediate = interview.get("immediate_eval", True)

            if is_immediate:
                navigate_to("Feedback")
            else:
                answers = interview.setdefault("answers", [])
                answers.append({
                    "number": question_index + 1,
                    "question": question_data["question"],
                    "answer": submitted_answer,
                    "score": None
                })
                
                if question_index + 1 >= total_questions:
                    if "vision_process" in st.session_state:
                        st.session_state.vision_process.terminate()
                        st.session_state.pop("vision_process", None)
                    navigate_to("Batch Evaluation")
                else:
                    interview["question_index"] = question_index + 1
                    save_draft(interview)
                    st.session_state.interview = interview
                    st.rerun()

        # --- CENTER BOTTOM ABORT SWITCH ---
        st.write("")
        abort_left, abort_mid, abort_right = st.columns([1, 1, 1])
        with abort_mid:
            if st.button("🚫 Abort interview", use_container_width=True):
                if "vision_process" in st.session_state:
                    st.session_state.vision_process.terminate()
                    st.session_state.pop("vision_process", None)
                
                delete_draft(current_user()["id"])
                st.session_state.pop("interview", None)
                st.session_state.pop("question_cache", None)
                navigate_to("Interview setup")


def feedback() -> None:
    interview = st.session_state.get("interview")
    answer = str(st.session_state.get("current_answer", ""))
    question = str(st.session_state.get("current_question", ""))
    if not interview or not question:
        navigate_to("Home")
        return

    feedback_key = hashlib.sha256(f"{question}\n{answer}".encode()).hexdigest()
    if st.session_state.get("feedback_key") != feedback_key:
        try:
            with st.spinner("AIM is assessing your answer…"):
                st.session_state.latest_feedback = evaluate_star_answer(
                    question=question,
                    answer=answer,
                    resume_context=get_or_load_resume_context(current_user()["id"]),
                )
            st.session_state.feedback_key = feedback_key
        except AIMAIError as exc:
            st.error(str(exc))
            return

    result = st.session_state.latest_feedback
    st.markdown('<p class="eyebrow">AIM ANALYZED YOUR ANSWER</p>', unsafe_allow_html=True)
    st.title(result["headline"])
    st.caption(result["summary"])
    a, b, c = st.columns(3)
    a.metric("Technical understanding", f"{result['scores']['technical_understanding']}/10")
    b.metric("Depth", f"{result['scores']['depth']}/10")
    c.metric("Relevance", f"{result['scores']['relevance']}/10")
    with st.container(border=True):
        st.markdown("**AIM's note**")
        st.write(result["aim_note"])

    curr_idx = interview.get("question_index", 0)
    
    # Resolve dynamic total questions (defaults to 10 if unparsed)
    total_questions = interview.get("total_questions", 10)
    if not total_questions:
        try:
            total_questions = int(interview.get("duration", "10"))
        except ValueError:
            total_questions = 10

    is_final_question = (curr_idx + 1 >= total_questions)

    # Navigation buttons
    if not is_final_question:
        if st.button("Next question", type="primary", use_container_width=True):
            with st.spinner("AIM is preparing your next question…"):
                # Save answer to state
                interview.setdefault("answers", []).append(
                    {"number": curr_idx + 1, "question": question, "answer": answer, "score": result["scores"]["overall"]}
                )
                interview["question_index"] = curr_idx + 1
                save_draft(interview)
                st.session_state.interview = interview
                
                # Pre-warm cache for the next question while spinner is active
                next_idx = curr_idx + 1
                next_cache_key = f"{interview.get('id', 'new')}:{next_idx}"
                question_cache = st.session_state.setdefault("question_cache", {})
                
                if next_cache_key not in question_cache:
                    resume_context = get_or_load_resume_context(current_user()["id"])
                    if resume_context:
                        try:
                            question_cache[next_cache_key] = generate_interview_question(
                                resume_context=resume_context,
                                interview_type=interview.get("type") or interview.get("interview_type", "Technical"),
                                difficulty=interview.get("difficulty", "Adaptive"),
                                previous_answers=interview["answers"],
                            )
                        except AIMAIError as exc:
                            st.error(str(exc))

                navigate_to("Live interview")
    else:
        st.info("You have completed all questions for this session!")

    if st.button(
        "🚫 Abort interview" if not is_final_question else "Complete & Save Session", 
        type="primary" if is_final_question else "secondary", 
        use_container_width=True
    ):
        all_answers = interview.setdefault("answers", [])
        if "vision_process" in st.session_state:
            st.session_state.vision_process.terminate()
            st.session_state.pop("vision_process", None)
            
        if not any(entry.get("question") == question for entry in all_answers):
            all_answers.append(
                {"number": curr_idx + 1, "question": question, "answer": answer, "score": result["scores"]["overall"]}
            )
        
        avg_score = round(sum(item["score"] for item in all_answers) / len(all_answers)) if all_answers else result["scores"]["overall"]
        
        with get_db() as conn:
            conn.execute(
                "INSERT INTO interviews(user_id, interview_type, score, duration) VALUES(?,?,?,?)",
                (
                    current_user()["id"], 
                    interview.get("type") or interview.get("interview_type", "Technical"), 
                    avg_score, 
                    str(total_questions)
                )
            )
        delete_draft(current_user()["id"])
        st.session_state.pop("interview", None)
        st.session_state.pop("question_cache", None)
        st.session_state.pop("feedback_key", None)
        st.session_state.pop("latest_feedback", None)
        navigate_to("Interviews")


def interviews() -> None:
    records = user_interviews(current_user()["id"])
    average = round(sum(record["score"] for record in records) / len(records)) if records else 0
    st.title("Your interviews")
    st.caption("Every answer gives AIM a sharper picture of you.")
    a, b, c = st.columns(3)
    a.metric("Total sessions", len(records))
    b.metric("Average score", f"{average}%")
    c.metric("Practice time", f"{len(records) * 20} min" if records else "0 min")
    st.subheader("Interview history")
    if records:
        st.dataframe(
            [{"Interview": record["interview_type"], "Date": record["created_at"], "Duration": record["duration"], "Score": f"{record['score']}%"} for record in records],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Your AIM journey starts here. Complete your first interview to see your history.")


def progress() -> None:
    records = user_interviews(current_user()["id"])
    readiness = round(sum(record["score"] for record in records) / len(records)) if records else 0
    st.markdown('<p class="eyebrow">THE BIGGER PICTURE</p>', unsafe_allow_html=True)
    st.title("Your progress")
    st.caption("Small practice sessions are adding up.")
    st.metric("Readiness score", f"{readiness} / 100", None if not records else f"{len(records)} completed interview(s)")
    st.line_chart({"Readiness": [0] if not records else [record["score"] for record in reversed(records)]})
    st.subheader("What AIM learned about you")
    if not records:
        st.info("Complete your first interview to unlock personalized insights.")
    else:
        for insight in [
            "AIM is learning how you explain concepts.",
            "Your next sessions will adapt to your answers.",
            "Complete more interviews to reveal your strongest and weakest areas.",
        ]:
            with st.container(border=True):
                st.write(insight)


def profile() -> None:
    user = current_user()
    st.markdown('<p class="eyebrow">YOUR AIM PROFILE</p>', unsafe_allow_html=True)
    st.title("Profile")
    st.write(f"**{user['name']}** · AI/ML Engineer")
    st.subheader("Core skills")
    
    skills = ["Python", "Machine Learning", "SQL", "OpenCV", "YOLO", "Statistics"]
    pills_html = "".join(f'<span class="skill-tag">{skill}</span>' for skill in skills)
    st.markdown(f'<div style="margin-bottom: 20px;">{pills_html}</div>', unsafe_allow_html=True)

    st.subheader("Project")
    with st.container(border=True):
        st.write("**Vehicle Detection System**")
        st.caption("Python · YOLO · OpenCV")


def resume() -> None:
    st.markdown('<p class="eyebrow">KNOW ME</p>', unsafe_allow_html=True)
    st.title("Build your AIM profile")
    st.caption("Upload your resume so AIM can understand your experience.")
    
    current_resume = get_user_resume_path(current_user()["id"])
    if current_resume:
        st.success(f"Active resume: `{current_resume.name}`")

    upload = st.file_uploader("Upload new resume (PDF or DOCX, max 10 MB)", type=["pdf", "docx"])
    if not upload:
        return
    if upload.size > MAX_UPLOAD_BYTES:
        st.error("This file is larger than 10 MB.")
        return

    upload_digest = hashlib.sha256(upload.getvalue()).hexdigest()
    if st.session_state.get("resume_upload_digest") == upload_digest:
        st.info("This resume is already loaded into AIM's private interview context.")
        return

    user_id = current_user()["id"]
    safe_name = f"user_{user_id}_{secrets.token_hex(8)}_{Path(upload.name).name}"
    destination = UPLOAD_DIR / safe_name
    destination.write_bytes(upload.getvalue())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO resumes(user_id,filename,stored_path) VALUES(?,?,?)",
            (user_id, upload.name, str(destination)),
        )
    st.success("Resume uploaded to private local storage.")
    try:
        with st.spinner("AIM is reading your experience…"):
            st.session_state.resume_context = ingest_resume(destination)
        st.session_state.resume_upload_digest = upload_digest
        st.caption("AIM has indexed your resume for personalized questions.")
    except AIMAIError as e:
        st.session_state.pop("resume_context", None)
        st.error(f"Failed to process resume: {e}")

def batch_evaluation() -> None:
    interview = st.session_state.get("interview")
    if not interview or not interview.get("answers"):
        navigate_to("Home")
        return

    st.markdown('<p class="eyebrow">FINAL COMPREHENSIVE ASSESSMENT</p>', unsafe_allow_html=True)
    st.title("Evaluating your entire interview…")
    
    resume_ctx = get_or_load_resume_context(current_user()["id"])
    all_answers = interview["answers"]
    
    if "batch_results" not in st.session_state:
        results = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        for idx, item in enumerate(all_answers):
            status_text.write(f"Grading Question {idx + 1} of {len(all_answers)}…")
            try:
                eval_data = evaluate_star_answer(
                    question=item["question"],
                    answer=item["answer"],
                    resume_context=resume_ctx
                )
                item["score"] = eval_data["scores"]["overall"]
                item["eval"] = eval_data
            except AIMAIError:
                item["score"] = 0
                item["eval"] = {"headline": "Evaluation failed", "summary": "N/A"}
            
            results.append(item)
            progress_bar.progress((idx + 1) / len(all_answers))
            
        st.session_state.batch_results = results

    results = st.session_state.batch_results
    avg_score = round(sum(r["score"] for r in results) / len(results)) if results else 0
    
    st.subheader(f"Overall Interview Score: {avg_score} / 100")
    
    for r in results:
        with st.expander(f"Question {r['number']}: {r['question']}"):
            st.write(f"**Your Answer:** {r['answer']}")
            if "eval" in r:
                st.write(f"**Headline:** {r['eval']['headline']}")
                st.write(f"**Feedback:** {r['eval'].get('aim_note', '')}")
                st.metric("Score", f"{r['score']}/100")

    if st.button("Complete & Save Session", type="primary", use_container_width=True):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO interviews(user_id, interview_type, score, duration) VALUES(?,?,?,?)",
                (current_user()["id"], interview.get("type") or interview.get("interview_type", "Technical"), avg_score, interview["duration"])
            )
        delete_draft(current_user()["id"])
        st.session_state.pop("interview", None)
        st.session_state.pop("batch_results", None)
        navigate_to("Interviews")

def settings() -> None:
    st.title("Settings")
    st.caption("Tune your mentoring experience.")
    st.toggle("Interview reminders", value=True)
    st.toggle("Voice mode", value=False)
    st.info("Production deployment should connect privacy, deletion, retention, and account controls to a backend policy service.")


# Application bootstrap
init_db()

if not current_user():
    auth_view()
    st.stop()

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

sidebar()

view_dispatch = {
    "Home": home,
    "Interviews": interviews,
    "Progress": progress,
    "Profile": profile,
    "Resume": resume,
    "Settings": settings,
    "Interview setup": interview_setup,
    "Live interview": live_interview,
    "Batch Evaluation": batch_evaluation,
    "Feedback": feedback,
}

render_view = view_dispatch.get(st.session_state.current_page, home)
render_view()