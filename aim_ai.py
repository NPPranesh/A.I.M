"""Local resume RAG and structured Llama 3.2 calls for Project A.I.M.

The public functions deliberately return JSON-serializable dictionaries so they
can be used by Streamlit today and moved to a FastAPI route without changing
their contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError


OLLAMA_URL = os.getenv("AIM_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("AIM_OLLAMA_MODEL", "llama3.2")
MAX_RESUME_CHARACTERS = 60_000
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180


class AIMAIError(RuntimeError):
    """A caller-safe error raised for invalid files or local model failures."""


class QuestionResponse(BaseModel):
    question: str = Field(min_length=12, max_length=420)
    focus: str = Field(min_length=2, max_length=120)
    difficulty: Literal["Beginner", "Adaptive", "Advanced"]


class StarDimension(BaseModel):
    situation: int = Field(ge=0, le=10)
    task: int = Field(ge=0, le=10)
    action: int = Field(ge=0, le=10)
    result: int = Field(ge=0, le=10)


class FeedbackScores(BaseModel):
    technical_understanding: int = Field(ge=0, le=10)
    depth: int = Field(ge=0, le=10)
    relevance: int = Field(ge=0, le=10)
    overall: int = Field(ge=0, le=100)


class FeedbackResponse(BaseModel):
    headline: str = Field(min_length=3, max_length=120)
    summary: str = Field(min_length=10, max_length=320)
    aim_note: str = Field(min_length=10, max_length=420)
    scores: FeedbackScores
    star: StarDimension
    strengths: list[str] = Field(min_length=1, max_length=3)
    improvements: list[str] = Field(min_length=1, max_length=3)


def ingest_resume(resume_path: str | Path) -> dict[str, Any]:
    """Extract a private PDF/DOCX and build a compact, JSON RAG context.

    The returned ``chunks`` are the user's local retrieval corpus. Persist this
    object server-side in production; in Streamlit it can be held in
    ``st.session_state.resume_context``.
    """
    path = Path(resume_path)
    if not path.is_file():
        raise AIMAIError("The saved resume file could not be found.")

    text = _extract_resume_text(path)
    text = _normalise_text(text)
    if len(text) < 80:
        raise AIMAIError(
            "AIM could not extract enough text. Upload a text-based PDF or DOCX, not a scanned image."
        )

    text = text[:MAX_RESUME_CHARACTERS]
    chunks = _chunk_text(text)
    if not chunks:
        raise AIMAIError("AIM could not build a retrieval index from this resume.")

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "resume_id": checksum[:20],
        "filename": path.name,
        "text": text,
        "chunks": chunks,
    }


def generate_interview_question(
    resume_context: dict[str, Any],
    interview_type: str,
    difficulty: str = "Adaptive",
    previous_answers: list[dict[str, Any]] | None = None,
    *,
    model: str = OLLAMA_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> dict[str, Any]:
    """Retrieve relevant resume excerpts and return one Llama-generated question."""
    _validate_resume_context(resume_context)
    normalised_difficulty = _normalise_difficulty(difficulty)
    previous_answers = previous_answers or []
    last_answer = _latest_answer(previous_answers)
    
    # 1. BUILD THE QUERY FIRST
    if not last_answer:
        # Drop 'interview_type' so the word "Technical" doesn't hijack the math.
        # Use specific engineering keywords found in Karan's resume.
        query = random.choice(["scalable", "migration", "database", "graphql", "infrastructure", "latency"])
    else:
        # For follow-ups, just search based on what the user actually said
        query = last_answer.strip()

    # 2. SEARCH THE RESUME SECOND
    evidence = retrieve_resume_evidence(resume_context, query, limit=4)
    
    prompt = f"""You are A.I.M., a rigorous but encouraging interview mentor.
Create exactly one {normalised_difficulty.lower()} {interview_type} interview question.
Ground it in the resume evidence below. If a prior answer is present, ask a
meaningful follow-up that tests a trade-off, decision, implementation detail,
or result. Ask only about work, tools, and constraints explicitly named in the
evidence. Do not introduce a hypothetical scenario or add a setting, dataset,
environment, feature, requirement, employer, achievement, technology, or number
that is absent from the evidence. Do not assume the candidate made a decision or
achieved a result unless the evidence says so. ``focus`` must be a short,
human-readable 2- to 6-word topic label, never a chunk ID. Do not include an
answer, a preamble, or more than one question.

RESUME EVIDENCE:
{_format_evidence(evidence)}

PRIOR ANSWER:
{last_answer or "No prior answer; begin with a strong resume-grounded question."}
"""
    payload = _ollama_structured(
        prompt=prompt,
        schema=QuestionResponse.model_json_schema(),
        model=model,
        ollama_url=ollama_url,
        temperature=0.25,
    )
    try:
        question = QuestionResponse.model_validate(payload)
    except ValidationError as exc:
        raise AIMAIError(f"Llama returned an invalid interview-question payload: {exc}") from exc

    result = question.model_dump()
    result["question"] = _single_question(result["question"])
    # Difficulty is an application decision, not a model decision.
    result["difficulty"] = normalised_difficulty
    result["resume_evidence"] = [
        {"chunk_id": chunk["id"], "text": chunk["text"]} for chunk in evidence
    ]
    return result


def evaluate_star_answer(
    question: str,
    answer: str,
    resume_context: dict[str, Any] | None = None,
    *,
    model: str = OLLAMA_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> dict[str, Any]:
    """Grade an answer with the STAR rubric and return validated JSON feedback."""
    if not question or not question.strip():
        raise AIMAIError("A question is required before an answer can be graded.")
    if not answer or not answer.strip() or answer.strip().lower() == "skipped":
        return _skipped_feedback()

    evidence: list[dict[str, Any]] = []
    if resume_context:
        _validate_resume_context(resume_context)
        evidence = retrieve_resume_evidence(resume_context, f"{question} {answer}", limit=3)

    prompt = f"""You are A.I.M., an exacting technical-interview evaluator.
Evaluate the candidate's answer to the stated question. Score the answer itself,
not the quality of the resume. Do not reward unsupported claims. Use the STAR
rubric even for technical answers: situation/context, task/goal, action/details,
and result/impact. It is valid to give low scores when a STAR element is absent.

SCORING:
- technical_understanding: correctness and command of the subject, 0-10
- depth: concrete implementation detail, reasoning, trade-offs, and evidence, 0-10
- relevance: directness in answering the question, 0-10
- overall: a fair aggregate score from 0-100

Write constructive, specific feedback. ``headline`` is a short encouraging
assessment, and ``summary`` is a one- or two-sentence explanation of the
assessment; neither may contain a standalone score or score-only wording.
``aim_note`` must name the most valuable next improvement in one or two
sentences. Never claim facts that are not in the answer or the resume excerpts.

QUESTION:
{question.strip()}

CANDIDATE ANSWER:
{answer.strip()}

OPTIONAL RESUME EVIDENCE (only to check consistency, never to fill gaps):
{_format_evidence(evidence) if evidence else "No resume context supplied."}
"""
    payload = _ollama_structured(
        prompt=prompt,
        schema=FeedbackResponse.model_json_schema(),
        model=model,
        ollama_url=ollama_url,
        temperature=0.15,
    )
    try:
        return FeedbackResponse.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise AIMAIError(f"Llama returned an invalid STAR-feedback payload: {exc}") from exc


def retrieve_resume_evidence(
    resume_context: dict[str, Any], query: str, limit: int = 4
) -> list[dict[str, Any]]:
    """A deterministic local lexical retrieval step over the resume chunks.

    It avoids sending the entire resume to Llama on every question. The selected
    excerpts then augment the generation/evaluation prompt (RAG).
    """
    _validate_resume_context(resume_context)
    terms = _terms(query)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, chunk in enumerate(resume_context["chunks"]):
        chunk_terms = _terms(chunk["text"])
        counts = Counter(chunk_terms)
        score = sum((1 + min(counts[term], 3)) for term in terms if term in counts)
        # A small deterministic tie-breaker gives stable output for reruns.
        scored.append((float(score), -position, chunk))

    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    chosen = [chunk for score, _, chunk in scored if score > 0][:limit]
    return chosen or resume_context["chunks"][:limit]


def _extract_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - setup issue
            raise AIMAIError("PDF support needs pypdf. Install it with: pip install pypdf") from exc
        try:
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        except Exception as exc:
            raise AIMAIError("AIM could not read this PDF. It may be encrypted or malformed.") from exc
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:  # pragma: no cover - setup issue
            raise AIMAIError("DOCX support needs python-docx. Install it with: pip install python-docx") from exc
        try:
            document = Document(str(path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as exc:
            raise AIMAIError("AIM could not read this DOCX file.") from exc
    raise AIMAIError("Only PDF and DOCX resumes are supported.")


def _chunk_text(text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_number = 1
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
        value = text[start:end].strip()
        if value:
            chunks.append({"id": f"resume-{chunk_number}", "text": value})
            chunk_number += 1
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _ollama_structured(
    *, prompt: str, schema: dict[str, Any], model: str, ollama_url: str, temperature: float
) -> dict[str, Any]:
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": temperature},
    }
    try:
        response = httpx.post(endpoint, json=body, timeout=90.0)
        response.raise_for_status()
        raw_response = response.json().get("response", "")
    except (httpx.HTTPError, ValueError) as exc:
        raise AIMAIError(
            "AIM could not reach local Ollama. Start it and ensure the configured model is installed."
        ) from exc
    return _json_object(raw_response)


def _json_object(raw_response: str) -> dict[str, Any]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AIMAIError("Llama did not return valid JSON. Try the request again.") from exc
    if not isinstance(parsed, dict):
        raise AIMAIError("Llama returned JSON in an unexpected format.")
    return parsed


def _normalise_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{1,}", value.lower()) if len(term) > 1]


def _single_question(value: str) -> str:
    """Keep a minor model formatting lapse from turning a UI title into a quiz."""
    value = re.sub(r"\s+", " ", value).strip()
    question_end = value.find("?")
    if question_end >= 0:
        return value[: question_end + 1]
    return f"{value.rstrip('.! ')}?"


def _format_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "No resume excerpts available."
    return "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in evidence)


def _normalise_difficulty(difficulty: str) -> Literal["Beginner", "Adaptive", "Advanced"]:
    values = {"beginner": "Beginner", "adaptive": "Adaptive", "advanced": "Advanced"}
    return values.get(difficulty.lower(), "Adaptive")


def _latest_answer(previous_answers: list[dict[str, Any]]) -> str:
    for item in reversed(previous_answers):
        answer = str(item.get("answer", "")).strip()
        if answer and answer.lower() != "skipped":
            return answer
    return ""


def _validate_resume_context(resume_context: dict[str, Any]) -> None:
    chunks = resume_context.get("chunks") if isinstance(resume_context, dict) else None
    if not isinstance(chunks, list) or not chunks or not all(
        isinstance(chunk, dict) and isinstance(chunk.get("id"), str) and isinstance(chunk.get("text"), str)
        for chunk in chunks
    ):
        raise AIMAIError("Resume context is missing or invalid. Upload and parse a resume first.")


def _skipped_feedback() -> dict[str, Any]:
    return FeedbackResponse(
        headline="No answer to assess yet",
        summary="AIM cannot assess technical understanding without a submitted answer.",
        aim_note="Try a concise answer with context, your specific action, and a measurable result.",
        scores=FeedbackScores(technical_understanding=0, depth=0, relevance=0, overall=0),
        star=StarDimension(situation=0, task=0, action=0, result=0),
        strengths=["You identified when to move on."],
        improvements=["Answer the next question with a concrete example."],
    ).model_dump()
