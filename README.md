# ABD's Interview Preparation Agent

An AI-powered interview coach: personalized study plans from your resume, voice-enabled mock interviews with adaptive coding rounds, honest per-answer scoring, live company-specific research, and a readiness debrief.

**Works with ANY major LLM provider** — Anthropic, OpenAI, or Google Gemini — switched by one line in `.env`.

## Architecture

```
Browser (static/index.html — thin view layer)
   │  POST /api/plan | /api/company_intel | /api/question | /api/answer | /api/review
   ▼
FastAPI (app.py — ALL agent logic and prompts)
   ├── LiteLLM        → Claude / GPT / Gemini / 100+ models (LLM_MODEL picks it)
   ├── pypdf          → resume PDF extracted locally (free, no tokens)
   └── Tavily API     → provider-neutral live web research (optional)
```

## Run locally

Requirements: Python 3.9+

```bash
# 1. Clone and install
git clone <your-repo-url>
cd Interview_Prep-by-ABD
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure — create .env (or copy .env.example) with TWO things:
#    the model, and the matching provider key
```

Your `.env` needs just two lines. Pick one provider:

| Provider | `.env` contents | Get a key at |
|---|---|---|
| Anthropic | `LLM_MODEL=claude-sonnet-4-6` + `ANTHROPIC_API_KEY=sk-ant-...` | platform.claude.com |
| OpenAI | `LLM_MODEL=gpt-4o` + `OPENAI_API_KEY=sk-...` | platform.openai.com |
| Google | `LLM_MODEL=gemini/gemini-2.5-pro` + `GEMINI_API_KEY=...` | aistudio.google.com |

Optionally add `TAVILY_API_KEY=tvly-...` (free tier at tavily.com) to enable live company research. Without it, everything else still works — you just won't get company-specific freshness.

```bash
# 3. Start
python app.py
# or: uvicorn app:app --reload --port 3001
```

Open **http://localhost:3001**. To switch providers later, edit `LLM_MODEL` and restart — nothing else changes.

## Features

- **Any LLM provider** — LiteLLM translates our prompts to whichever API `LLM_MODEL` names; the matching key is read from the environment automatically. You can even use a cheap model day-to-day (`gpt-4o-mini`, `claude-haiku-4-5-20251001`, `gemini/gemini-2.5-flash`) and a strong one before the real interview.
- **PDF resume upload** — extracted locally with pypdf: instant, free, and identical across providers. Scanned/image-only PDFs can't be extracted — paste text instead (the app tells you if so).
- **Live company research (optional)** — the backend runs 2 Tavily web searches (recent interview experiences, round structure, hot topics, company news), then the LLM distills them into intel that biases every question. Fetched once per session. Tavily's free tier (~1,000 searches/month) is far more than you'll need.
- **Voice interviews** — questions spoken aloud, answers via microphone (browser Web Speech API; best in Chrome/Edge, requires HTTPS or localhost).
- **Adaptive coding rounds** — included automatically for technical roles with a correctness/complexity/readability rubric; skipped for non-technical roles.
- **Mid-interview navigation** — a ⌂ Home button returns to the plan screen, and a round dropdown in the toolbar switches focus (e.g. coding → technical) from the next question without losing your transcript.
- **Keys stay server-side** — the browser never sees any API key.

## Deploy

Set the env vars (`LLM_MODEL`, the provider key, optionally `TAVILY_API_KEY`) on the host — never commit `.env` (it's gitignored).

**Render / Railway:** new Web Service from your GitHub repo → build `pip install -r requirements.txt` → start `uvicorn app:app --host 0.0.0.0 --port $PORT` → add the env vars.

**VPS:** same uvicorn command behind Nginx/Caddy for HTTPS (the mic requires a secure context).

## Notes

- Prompts were tuned on Claude; other models may grade slightly differently or occasionally drift from JSON formatting (the parser is defensive, but if one model misbehaves persistently, tighten the "Respond ONLY with JSON" lines in `app.py`).
- All prompts live in `app.py` — that's the file to edit to change the agent's behavior.
- State lives in the browser tab; refresh resets the session. Add SQLite persistence in `app.py` for score history.
- Typical cost per full mock session: well under $0.50 on any mid-tier model; Tavily research is free-tier.
