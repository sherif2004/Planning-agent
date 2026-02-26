# ⚡ SQL Agent — Natural Language to PostgreSQL

> Ask your database anything in plain English. Powered by an LLM (via OpenRouter), FastAPI, and SQLAlchemy.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)
![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E?style=flat&logo=railway&logoColor=white)

---

## ✨ Features

- 🧠 **LLM-powered query planner** — converts plain English into a structured JSON plan, then into safe SQL
- 🛡️ **Validation layer** — every column and table is checked against the real schema before execution
- 📊 **Beautiful UI** — dark terminal-style interface with tabbed results (Data / SQL / Plan)
- 🔒 **API key auth** — simple `X-API-Key` header protection
- ⚡ **Connection pooling** — production-ready SQLAlchemy engine
- 📖 **Auto docs** — Swagger UI at `/docs`, ReDoc at `/redoc`

---

## 🗂️ Project Structure

```
.
├── main.py            # FastAPI app — all agent logic
├── index.html         # Frontend UI (served at /)
├── requirements.txt   # Python dependencies
├── Procfile           # Railway / Heroku start command
├── railway.toml       # Railway config (optional)
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/sql-agent.git
cd sql-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file (never commit this):

```env
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/your_db
OPENROUTER_API_KEY=sk-or-v1-...
APP_API_KEY=your-secret-key
LLM_MODEL=openai/gpt-4o-mini
```

### 4. Run the server

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** — the UI will load automatically.

---

## 🌐 Deploy on Railway

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sql-agent.git
git push -u origin main
```

### Step 2 — Create a Railway project

1. Go to [railway.app](https://railway.app) and sign in
2. Click **New Project → Deploy from GitHub repo**
3. Select your repository

### Step 3 — Add a PostgreSQL database

1. In your Railway project, click **+ New → Database → Add PostgreSQL**
2. Railway will automatically inject `DATABASE_URL` into your service — no manual copy needed

### Step 4 — Set environment variables

In your Railway service → **Variables** tab, add:

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` |
| `APP_API_KEY` | `your-secret-key` |
| `LLM_MODEL` | `openai/gpt-4o-mini` |
| `PORT` | `8000` *(Railway sets this automatically)* |

> ⚠️ Do **not** set `DATABASE_URL` manually — Railway injects it from the linked Postgres service.

### Step 5 — Add a Procfile

Create a file named `Procfile` (no extension) in your project root:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Step 6 — Deploy

Railway auto-deploys on every push to `main`. After the build completes:

- Your app will be live at `https://your-app.up.railway.app`
- UI → `https://your-app.up.railway.app/`
- Swagger docs → `https://your-app.up.railway.app/docs`
- Health check → `https://your-app.up.railway.app/health`

---

## 📡 API Reference

### `POST /query`

Ask a natural language question about your database.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-secret-key
```

**Request body:**
```json
{
  "question": "Which department has the highest total salary expense?"
}
```

**Response:**
```json
{
  "question": "Which department has the highest total salary expense?",
  "plan": { ... },
  "sql": "SELECT \"department\", SUM(\"salary\") AS \"total_salary\" FROM ...",
  "rows": [{ "department": "Engineering", "total_salary": 520000.0 }],
  "answer": "The Engineering department has the highest total salary expense at $520,000."
}
```

### `GET /health`

Returns DB connection status. Used by Railway health checks.

```json
{ "status": "ok", "db": "ok" }
```

### `GET /schema`

Returns the database schema (tables and columns).

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string |
| `OPENROUTER_API_KEY` | ✅ | — | Your OpenRouter API key |
| `APP_API_KEY` | ✅ | `change-me-in-production` | API key for endpoint auth |
| `LLM_MODEL` | ❌ | `openai/gpt-4o-mini` | Model to use via OpenRouter |
| `PORT` | ❌ | `8000` | Port (Railway sets automatically) |

---

## 🛡️ Security Notes

- Only `SELECT` statements are ever executed — writes are blocked at the code level
- Every column and table name is validated against the live schema before SQL is built — no prompt injection into queries
- API key is required for all `/query` and `/schema` endpoints
- Never commit `.env` or hardcode secrets — use Railway's Variables tab

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM Provider | OpenRouter (GPT-4o-mini) |
| Database ORM | SQLAlchemy 2.0 |
| Database | PostgreSQL |
| Server | Uvicorn |
| Hosting | Railway |

---

## 📄 License

MIT — free to use, modify, and deploy.
