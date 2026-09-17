# AI Campus Helpdesk

A Retrieval-Augmented Generation (RAG) chatbot with a multi-agent architecture for
answering campus-related questions, built as an intermediate-level academic project.

> **Important:** This project uses **Greenfield Institute of Technology (GIT)**, a
> completely **fictional** sample university created for this project. All policies,
> fees, and contact details in `knowledge_base/` are sample data, not real information
> from any actual institution.

## 0. Windows Quick Start (for complete beginners)

If you received this project as a zip file and are on Windows, follow these exact
steps in order. No prior experience needed.

**1. Unzip the project**
Right-click the zip file → "Extract All" → pick a simple location like
`C:\ai-campus-helpdesk`.

**2. Install Python**
- Go to https://www.python.org/downloads/ and download Python 3.11 or newer.
- Run the installer. **Important:** check the box "Add python.exe to PATH" at the
  bottom of the first screen before clicking Install.
- Verify it worked: open **Command Prompt** (search "cmd" in the Start menu) and type:
  ```
  python --version
  ```
  It should print something like `Python 3.12.x`.

**3. Install Ollama (runs the AI model locally)**
- Go to https://ollama.com/download and download the Windows installer.
- Run it — it installs and starts running in the background automatically (look for
  its icon in the system tray).

**4. Download an AI model**
Open Command Prompt and run:
```
ollama pull llama3.2:3b
```
This downloads about 2GB and may take a few minutes.

**5. Open the project folder in Command Prompt**
```
cd C:\ai-campus-helpdesk
```
(adjust the path to wherever you extracted the zip)

**6. Create a virtual environment**
```
python -m venv .venv
.venv\Scripts\activate
```
Your prompt should now show `(.venv)` at the start of the line.

**7. Install the required packages**
```
pip install -r requirements.txt
```
This takes a few minutes (it downloads several libraries including PyTorch).

**8. Create the settings file**
```
copy .env.example .env
```
No editing needed — the defaults work out of the box.

**9. Build the knowledge database**
```
python scripts\build_index.py
```
You should see progress messages ending in something like
"Collection now contains 72 items."

**10. Start the backend** (keep this window open)
```
uvicorn app.main:app --reload
```
Wait for the line `Uvicorn running on http://0.0.0.0:8000`.

**11. Start the frontend — in a NEW Command Prompt window**
```
cd C:\ai-campus-helpdesk
.venv\Scripts\activate
streamlit run ui\streamlit_app.py
```
This should automatically open your browser to the chat page. If not, go to
`http://localhost:8501` manually.

**12. Try it out**
Type a question like *"What documents are required for hostel admission?"* and click
Ask. The first answer will be slow (30 seconds to a few minutes depending on your
CPU) since it's running a local AI model — that's normal, not a bug.

### Common snags
- **"python is not recognized"** → Python wasn't added to PATH; reinstall and check
  that box on the first installer screen.
- **Answers are very slow** → normal for CPU-only local models; be patient on the
  first question especially.
- **Streamlit says "Could not connect to the backend API"** → the `uvicorn` window
  from step 10 must stay open in a separate window while you use the app.
- **Antivirus/firewall popup** when starting Ollama or uvicorn → click Allow, it's
  just opening a local network port on your own machine.

The rest of this README (below) covers the same steps in more technical detail,
plus things like adding new knowledge-base documents and cloud deployment — useful
once you're comfortable with the basics above.

## 1. What this project is

A student can type a question like *"What documents are required for hostel
admission?"* into a Streamlit web page. The system:

1. Routes the question to the right department (Academic, Hostel, Administration, or
   Student Services) using an LLM-based **Router Agent**.
2. Hands the question to that department's specialized **Agent**.
3. Retrieves the most relevant chunks of text from that department's knowledge base
   using a **RAG (Retrieval-Augmented Generation)** pipeline backed by ChromaDB.
4. Passes the retrieved text to a local LLM (via **Ollama**) with a strict prompt that
   forbids inventing information.
5. Returns a grounded answer, plus the department, agent, and source documents used.

If the answer isn't in the knowledge base, the system says so instead of guessing.

## 2. Architecture

```
Student
  |
  v
Streamlit UI (ui/streamlit_app.py)
  |  HTTP POST /ask
  v
FastAPI backend (app/main.py)
  |
  v
Router Agent (app/agents/router.py)  --LLM classification, keyword fallback--
  |
  v
Department Agent (Academic / Hostel / Administration / Student Services)
  |
  v
RAG retrieval (app/rag/retriever.py) -> ChromaDB (department-filtered search)
  |
  v
LLM (app/llm/ollama_client.py) -> Ollama (local) or a cloud provider
  |
  v
Grounded answer + department + sources
  |
  v
Student
```

## 3. Technologies used

| Purpose             | Technology                                   |
|----------------------|-----------------------------------------------|
| Backend API          | FastAPI + Pydantic                            |
| Frontend             | Streamlit                                     |
| Orchestration/RAG glue | LangChain (text splitting)                  |
| Vector database       | ChromaDB (persistent, local)                |
| Embeddings            | sentence-transformers (`all-MiniLM-L6-v2`)  |
| LLM inference (local)| Ollama (e.g. `llama3.2:3b`, configurable)     |
| PDF parsing           | pypdf                                       |
| Config                | python-dotenv                               |
| Testing               | pytest                                      |

## 4. Folder structure

```
ai-campus-helpdesk/
  app/
    main.py                  FastAPI app (POST /ask, GET /health)
    agents/                  Router Agent + 4 department agents
    rag/                     document loading, chunking, embeddings, vector store, retriever
    llm/                     LLM abstraction layer (Ollama / cloud) + prompt templates
    models/                  Pydantic request/response schemas
    utils/                   config.py (env vars), logger.py
  knowledge_base/            10 sample .txt documents (fictional university)
  evaluation/
    questions.json           49 evaluation questions
    evaluate.py               runs Experiments A/B/C and writes CSV results
    results/                  CSV + summary.json output (generated)
  ui/
    streamlit_app.py         Streamlit frontend
  data/chroma/                ChromaDB persistent storage (generated)
  scripts/
    build_index.py           builds/rebuilds the vector index
  tests/                     pytest test suite
  requirements.txt
  .env.example
  research_paper.txt
  summary.txt
```

## 5. Installation

### 5.1 Prerequisites

- Python 3.11+ (this project was built and tested on Python 3.14)
- [Ollama](https://ollama.com) installed locally (for local LLM inference)

### 5.2 Install Ollama

Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
macOS: download from https://ollama.com/download
Windows: download the installer from https://ollama.com/download

Start the Ollama server (if not already running as a service):
```bash
ollama serve
```

### 5.3 Pull a model

```bash
ollama pull llama3.2:3b
```
You can use any model you have pulled — just set `OLLAMA_MODEL` in `.env` to match
(e.g. `qwen2.5:3b`, `llama3.1:8b` for better quality if your machine can handle it).

### 5.4 Create a Python virtual environment

```bash
cd ai-campus-helpdesk
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
```

### 5.5 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.6 Configure environment variables

```bash
cp .env.example .env
```
The defaults work out of the box for local Ollama usage. Edit `.env` only if you want
to change the model, ports, or chunking parameters.

## 6. Build the vector database

Before first use (and any time you add/change knowledge base documents):

```bash
python scripts/build_index.py
```

This loads every file in `knowledge_base/`, splits it into chunks, embeds them, and
stores them in ChromaDB at `data/chroma/`. You should see progress logs and a final
"Collection now contains N items" message.

## 7. Start the backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check it's running:
```bash
curl http://localhost:8000/health
```

## 8. Start the frontend (Streamlit)

In a **second terminal** (keep the backend running):
```bash
source .venv/bin/activate
streamlit run ui/streamlit_app.py
```
This opens the app in your browser, usually at `http://localhost:8501`.

## 9. Running the evaluation

With the backend's dependencies importable (no need for the server to be running,
since the evaluation script calls the agents directly) and Ollama running:

```bash
python evaluation/evaluate.py
```

This runs three experiments (see `research_paper.txt` for full methodology):
- **A**: Plain LLM, no RAG
- **B**: LLM + RAG, no department routing
- **C**: Full system — Multi-Agent Router + RAG (this is the main system)

Results are written to `evaluation/results/*.csv` and a combined
`evaluation/results/summary.json`.

Note: because this project uses a small (3B parameter) CPU-run local model by default,
running the full 49-question evaluation across 3 experiments can take a long time
(each generation can take 30s-3 minutes on CPU-only hardware; a full run of all three
experiments took roughly 2 hours in this project's own test environment). Use
`python evaluation/evaluate.py --experiment c` to test only the main system, or add
`--limit N` to any experiment to run on just the first N questions.

Real results from a completed run (llama3.2:3b, CPU-only, 2026-08-16) are already
included in this repository — see `research_paper.txt` Section 21 for the full
discussion, or `evaluation/results/summary.json` / the CSV files for raw numbers.
Headline finding: RAG improved answer accuracy from 28.9% (plain LLM) to 92.9%
(LLM+RAG) on a matched 15-question subset; the full multi-agent system scored 66.9%
answer accuracy and 77.6% routing accuracy on all 49 questions — see the research
paper for why routing did not further improve on plain RAG in this run.

## 10. Running tests

```bash
pytest tests/ -v
```

Some retrieval tests are skipped automatically if the vector index hasn't been built
yet (`python scripts/build_index.py` first). API and router tests do not require
Ollama to be running (they use mocking/keyword fallback).

## 11. Adding new knowledge-base documents

1. Add a `.txt` or `.pdf` file to `knowledge_base/`.
2. If it belongs to a new filename, add an entry to `FILE_DEPARTMENT_MAP` in
   `app/rag/document_loader.py` so it's tagged with the right department (files not
   listed default to "Student Services").
3. Rebuild the index: `python scripts/build_index.py`.

## 12. Cloud deployment

The backend is not hard-wired to Ollama. `app/llm/ollama_client.py` implements an
abstraction: set `LLM_PROVIDER=cloud` in `.env` along with `CLOUD_LLM_API_KEY`,
`CLOUD_LLM_BASE_URL`, and `CLOUD_LLM_MODEL` to use any OpenAI-compatible hosted API
instead of local Ollama — no code changes required, only environment variables.
**Never commit real API keys**; `.env` is git-ignored, only `.env.example` (with
placeholders) is committed.

Practical deployment approach for this project:
1. **Backend**: containerize `app/` with a simple Dockerfile (not included by
   default, to keep the project simple — add one if deploying) and deploy to any
   container host (Render, Railway, Fly.io, a cloud VM, etc.). Set `LLM_PROVIDER=cloud`
   and the cloud credentials as environment variables/secrets on the host.
2. **Vector store**: ChromaDB's persistent client writes to a local directory
   (`data/chroma/`). For cloud deployment, mount a persistent volume, or run
   `scripts/build_index.py` as part of the deployment/startup process to rebuild it
   from `knowledge_base/` each time (cheap for this project's small dataset).
3. **Frontend**: deploy `ui/streamlit_app.py` on Streamlit Community Cloud or the
   same host as the backend, pointing `API_URL` at the deployed backend's public URL.

This project has **not** been deployed to any live cloud service as part of this
build — the code, configuration, and instructions above are prepared for deployment,
but no deployment was actually performed or claimed.

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not connect to Ollama` | Run `ollama serve` in a terminal and keep it running. |
| Model errors / "is the model pulled?" | Run `ollama pull <model_name>` matching `OLLAMA_MODEL` in `.env`. |
| `Vector store missing` / empty answers with no sources | Run `python scripts/build_index.py`. |
| Streamlit says "Could not connect to the backend API" | Make sure `uvicorn app.main:app` is running in another terminal on the port matching `API_PORT`. |
| Answers are very slow | Small CPU-only models can take 30s-3 minutes per answer. Use a smaller/faster model, or enable GPU support in Ollama if available. |
| `pip install` fails on some package | Ensure you're using Python 3.11+ and a fresh virtual environment. |
| Import errors when running scripts directly | Run scripts from the project root (e.g. `python scripts/build_index.py`, not from inside `scripts/`). |

## 14. Known limitations (see `research_paper.txt` for full discussion)

- The LLM-based semantic router can occasionally misclassify a question when using a
  very small (3B parameter) local model; the keyword-based fallback exists to catch
  outright LLM failures, but does not fully correct semantic misclassifications by a
  working-but-imperfect LLM router.
- Answer correctness is evaluated with a simple keyword/concept-overlap check, not a
  full semantic evaluator — see `evaluation/evaluate.py` docstring for details.
- This is a prototype/research system, not a production-ready university platform.
