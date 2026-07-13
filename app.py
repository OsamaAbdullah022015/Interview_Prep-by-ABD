"""
ABD's Interview Preparation Agent — v3, provider-agnostic backend.

Works with ANY major LLM provider:
  - the brain:   LiteLLM (`LLM_MODEL` in .env picks the provider — Anthropic,
                 OpenAI, Google Gemini, and 100+ others)
  - the resume:  extracted locally with pypdf (free, no LLM tokens, any provider)
  - the web:     Tavily search API (provider-neutral live company research)

All the agent's intelligence lives in this file: the prompts for each stage
(plan / question / evaluation / debrief) and the endpoints the UI calls.
The frontend (static/index.html) is a thin view layer.

Run:  uvicorn app:app --reload --port 3001    (or: python app.py)
"""

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

import litellm  # noqa: E402  (import after load_dotenv so it sees the keys)
from litellm import completion  # noqa: E402

litellm.drop_params = True        # silently drop params a provider doesn't support
litellm.suppress_debug_info = True

MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ABD's Interview Preparation Agent")


# ---------------------------------------------------------------------------
# LLM helpers (provider-agnostic via LiteLLM)
# ---------------------------------------------------------------------------

def ask_llm(prompt: str, max_tokens: int = 1024) -> str:
    """One call, any provider. LiteLLM reads the matching key from the env
    (ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / ...) based on MODEL."""
    resp = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        reasoning_effort="none"
    )
    return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in markdown fences — strip and parse defensively."""
    clean = re.sub(r"```json|```", "", text).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end != -1:
        clean = clean[start : end + 1]
    return json.loads(clean)


def error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Local PDF extraction (no LLM involved — free and provider-neutral)
# ---------------------------------------------------------------------------

def extract_pdf_text(b64: str) -> str:
    reader = PdfReader(io.BytesIO(base64.b64decode(b64)))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 80:
        raise RuntimeError(
            "Couldn't extract text from this PDF — it may be a scanned image. "
            "Please paste your resume as text instead."
        )
    return text[:8000]  # keep prompts lean


# ---------------------------------------------------------------------------
# Web search (Tavily — provider-neutral)
# ---------------------------------------------------------------------------

def tavily_search(query: str, max_results: int = 5) -> list:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at tavily.com and add it "
            "to .env to enable live company research."
        )
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "max_results": max_results, "search_depth": "advanced"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def format_search_results(results: list) -> str:
    seen, lines = set(), []
    for res in results:
        url = res.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        lines.append(
            f"SOURCE: {res.get('title', '')} ({url})\n{res.get('content', '')[:900]}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Profile(BaseModel):
    domain: str
    role: str
    exp: str
    weeks: str
    company: str = ""  # optional target company — enables live web research


class TranscriptItem(BaseModel):
    role: str  # "coach" | "you"
    text: str
    category: Optional[str] = None
    score: Optional[int] = None


class PlanRequest(BaseModel):
    profile: Profile
    resume_text: str = ""
    resume_pdf_b64: str = ""


class QuestionRequest(BaseModel):
    profile: Profile
    resume_text: str = ""
    focus: str = "Full mock (mixed)"
    transcript: List[TranscriptItem] = []
    company_intel: str = ""


class AnswerRequest(BaseModel):
    profile: Profile
    resume_text: str = ""
    focus: str
    transcript: List[TranscriptItem] = []
    question: str
    category: str
    answer: str
    company_intel: str = ""


class ReviewRequest(BaseModel):
    profile: Profile
    resume_text: str = ""
    focus: str
    transcript: List[TranscriptItem] = []
    company_intel: str = ""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def profile_text(p: Profile, resume_text: str) -> str:
    company_line = f"- Target company: {p.company}\n" if p.company else ""
    return (
        "Candidate profile:\n"
        f"- Domain: {p.domain}\n"
        f"- Target role: {p.role}\n"
        + company_line
        + f"- Experience: {p.exp}\n"
        f"- Prep timeline: {p.weeks} weeks\n"
        f"- Resume:\n{resume_text or '(not provided)'}"
    )


def transcript_text(items: List[TranscriptItem]) -> str:
    lines = []
    for t in items:
        if t.role == "coach":
            lines.append(f"Interviewer asked ({t.category}): {t.text}")
        else:
            scored = f" (scored {t.score}/10)" if t.score else ""
            lines.append(f"Candidate answered: {t.text}{scored}")
    return "\n".join(lines) or "(none yet)"


def intel_block(company_intel: str) -> str:
    if not company_intel:
        return ""
    return (
        "\nLive company research (fetched from the web at session start):\n"
        f"{company_intel}\n"
        "Bias questions toward the hot_topics and the company's actual round "
        "structure. You may adapt sample_questions but don't copy them verbatim.\n"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/plan")
def make_plan(req: PlanRequest):
    """Extract the resume (locally, if a PDF was uploaded) and build the plan."""
    try:
        resume_text = req.resume_text.strip()
        if req.resume_pdf_b64 and not resume_text:
            resume_text = extract_pdf_text(req.resume_pdf_b64)

        n_weeks = min(int(req.profile.weeks or 4), 6)
        raw = ask_llm(
            "You are an expert interview coach. "
            + profile_text(req.profile, resume_text)
            + "\n\nCreate a concise interview prep plan. Be terse — short phrases, "
            "no filler. If the role is technical (software/data/ML/etc.), the plan "
            "MUST include coding practice (DSA or role-relevant problems); if "
            "non-technical, do not include coding. Respond ONLY with JSON, no "
            "markdown, exactly this shape:\n"
            '{"gap_analysis":["3-4 short gaps: resume vs target role"],'
            '"focus_areas":["4-5 short skill areas to drill"],'
            '"weeks":[{"week":1,"theme":"...","tasks":["3 short tasks"]}]}\n'
            f"Include exactly {n_weeks} weeks.",
            max_tokens=1800,
        )
        return {"resume_text": resume_text, "plan": parse_json(raw)}
    except Exception as exc:  # noqa: BLE001
        return error_response(exc)


@app.post("/api/company_intel")
def company_intel(req: QuestionRequest):
    """
    Live company research: our code fetches the web via Tavily, then ANY LLM
    synthesizes it. Called ONCE per session, not per question.
    """
    try:
        c = req.profile.company.strip()
        if not c:
            return {"intel": None}

        results = tavily_search(
            f"{c} {req.profile.role} interview process experience questions", 5
        )
        results += tavily_search(
            f"{c} interview questions {req.profile.domain} recent", 4
        )
        sources = format_search_results(results)
        if not sources:
            raise RuntimeError(f"No web results found for {c}.")

        raw = ask_llm(
            f"Below are fresh web search results about interviews at {c} for a "
            f"{req.profile.role} ({req.profile.domain}, {req.profile.exp} "
            "experience).\n\n"
            f"{sources}\n\n"
            "Using ONLY the information above, summarize the company's CURRENT "
            "interview process. Output ONLY JSON, no markdown, exactly:\n"
            '{"process":"1-2 sentences on the rounds",'
            '"hot_topics":["5-8 topics currently being asked"],'
            '"sample_questions":["4-6 recently reported questions"],'
            '"company_context":"1-2 sentences of recent news relevant in interviews"}',
            max_tokens=1200,
        )
        return {"intel": parse_json(raw)}
    except Exception as exc:  # noqa: BLE001
        return error_response(exc)


@app.post("/api/question")
def next_question(req: QuestionRequest):
    """Ask the next interview question, adapted to profile, focus, and history."""
    try:
        raw = ask_llm(
            profile_text(req.profile, req.resume_text)
            + "\n\nMock interview so far:\n"
            + transcript_text(req.transcript)
            + f"\n\nInterview focus: {req.focus}\n"
            + intel_block(req.company_intel)
            + "\nAsk the next interview question. Rules:\n"
            "- Match difficulty to the candidate's experience level.\n"
            "- Vary categories, don't repeat the same category back-to-back.\n"
            "- Probe things from the resume when possible.\n"
            "- If the domain/role is technical, include coding questions regularly: "
            "small self-contained problems solvable by typing a function (state "
            "input/output and an example). If the role is clearly non-technical, "
            "never use the coding category.\n"
            "- If a previous answer was weak, you may follow up on it.\n"
            'Respond ONLY with JSON: {"question":"...","category":"behavioral|'
            'technical|coding|system design|resume deep-dive|situational"}',
            max_tokens=800,
        )
        return parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        return error_response(exc)


@app.post("/api/answer")
def evaluate_answer(req: AnswerRequest):
    """Score the candidate's answer like a real interviewer debriefing."""
    try:
        coding_rubric = ""
        if req.category == "coding":
            coding_rubric = (
                " Since this is a coding question, judge correctness first, then edge "
                "cases, time/space complexity, and readability. Mention the complexity "
                "of their approach and whether a better one exists."
            )
        raw = ask_llm(
            profile_text(req.profile, req.resume_text)
            + "\n\nMock interview so far:\n"
            + transcript_text(req.transcript)
            + f"\n\nInterview focus: {req.focus}\n"
            + intel_block(req.company_intel)
            + f'\nThe interviewer just asked ({req.category}): "{req.question}"\n'
            f'The candidate answered:\n"""\n{req.answer}\n"""\n\n'
            "Evaluate honestly, like a real interviewer debriefing — don't "
            "inflate scores." + coding_rubric + " Respond ONLY with JSON: "
            '{"score":1-10,"feedback":"2-3 sentences on what worked and what '
            'didn\'t","tip":"1 concrete sentence on how a stronger answer would '
            'sound"}',
            max_tokens=800,
        )
        return parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        return error_response(exc)


@app.post("/api/review")
def final_review(req: ReviewRequest):
    """End-of-session debrief: readiness score, strengths, gaps, next drills."""
    try:
        raw = ask_llm(
            profile_text(req.profile, req.resume_text)
            + "\n\nMock interview so far:\n"
            + transcript_text(req.transcript)
            + f"\n\nInterview focus: {req.focus}\n"
            + intel_block(req.company_intel)
            + "\nThe mock interview is over. Give a final debrief. Respond ONLY "
            'with JSON: {"readiness":0-100,"summary":"2 sentences",'
            '"strengths":["2-3 short items"],"improve":["2-3 short items"],'
            '"next_steps":["3 short drill suggestions"]}',
            max_tokens=800,
        )
        return parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        return error_response(exc)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 3000)), reload=True)
