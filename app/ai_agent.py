"""
Member 5 — Cognitive Architect
--------------------------------
This module is the "brain" of Project A.I.M. It cures the LLM's amnesia
by retrieving the candidate's resume (Retrieval), injecting it into the
prompt alongside their live answer and eye-contact score (Augmentation),
and asking Llama 3.2 to produce a grounded follow-up question (Generation).

Three "state" variables feed every decision the model makes:
    1. Historical State  -> resume text pulled from the candidate's PDF
    2. Linguistic State   -> the transcript of what they just said
    3. Physical State     -> the eye-contact score from the vision pipeline
"""

from functools import lru_cache
from ollama import AsyncClient
from pypdf import PdfReader
import random

# ---------------------------------------------------------------------------
# 1. RETRIEVAL — pull the candidate's resume off disk and cache it.
#    A resume is small enough to hand over in full (like a lawyer handing a
#    judge the entire exhibit), so we don't need a vector DB / chunking here.
#    lru_cache means we only ever parse a given PDF once per server run,
#    even though evaluate_and_respond() may call this every single turn.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def load_resume_text(pdf_path: str) -> str:
    """Extracts plain text from a candidate's resume PDF. Returns a safe
    fallback string instead of raising, so a missing/corrupt resume never
    crashes an interview session."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text = text.strip()
        return text if text else "(Resume file was empty or unreadable.)"
    except FileNotFoundError:
        return "(No resume was provided for this candidate.)"
    except Exception as e:
        return f"(Resume could not be parsed: {e})"


# ---------------------------------------------------------------------------
# 2. AUGMENTATION — build the system prompt as explicit "state logic".
#    Kept separate from generate_follow_up() so it's easy to unit-test or
#    show on a slide by itself.
# ---------------------------------------------------------------------------
def build_system_prompt(resume_context: str, eye_contact: int) -> str:
    eye_contact_instruction = (
        "Their eye contact is LOW. Open with one short, encouraging line "
        "reminding them to look at the camera, then continue."
        if eye_contact < 70
        else "Their eye contact is fine — do not mention it at all."
    )

    return f"""You are Alex, a Senior Software Engineer conducting a live technical mock interview.

You will be given three pieces of state. Weigh them in this order:

[1. HISTORICAL STATE — candidate's resume]
\"\"\"
{resume_context}
\"\"\"

[2. LINGUISTIC STATE — what they just said out loud]
(provided in the user message)

[3. PHYSICAL STATE — eye contact score: {eye_contact}%]
{eye_contact_instruction}

RULES:
- Ask exactly ONE challenging technical follow-up question.
- Prefer grounding the question in a specific detail from the resume that
  connects to what they just said (e.g. a tool, company, or project named
  in the resume). If nothing connects, ask a solid follow-up on the answer
  alone — never invent resume details that aren't there.
- Use the spirit of the STAR method (Situation, Task, Action, Result) to
  probe for a concrete detail they glossed over.
- Tone: conversational and direct, like a real interviewer, not a chatbot.
- Output ONLY what you would say out loud. No labels, no markdown, no
  preamble like "Here's a question:".
- Hard limit: 30 words or fewer, including the eye-contact reminder if any.
"""


# ---------------------------------------------------------------------------
# 3. GENERATION — call the local Llama 3.2 model via Ollama.
# ---------------------------------------------------------------------------
async def generate_follow_up(
    candidate_answer: str,
    eye_contact: int,
    resume_context: str = "(No resume was provided for this candidate.)",
) -> str:
    """Uses a local Llama model via Ollama to generate a grounded,
    context-aware push-back question."""

    system_prompt = build_system_prompt(resume_context, eye_contact)

    print(f"\n[Ollama] Local AI is thinking... (Eye Contact: {eye_contact}%)")

    response = await AsyncClient().chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": candidate_answer},
        ],
        options={
            "temperature": 0.4,  
            "num_predict": 80, 
        },
    )

    return response["message"]["content"].strip()
