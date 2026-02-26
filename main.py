import os
import json
import logging
from decimal import Decimal
from datetime import date, datetime
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import uvicorn

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sql-agent")

# ─────────────────────────────────────────────
# Config (env-driven)
# ─────────────────────────────────────────────
DATABASE_URL       = os.getenv("DATABASE_URL",       "postgresql+psycopg2://postgres:2004@localhost:5432/company_db")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
APP_API_KEY        = os.getenv("APP_API_KEY",        "change-me-in-production")   # simple bearer key
LLM_MODEL          = os.getenv("LLM_MODEL",          "openai/gpt-4o-mini")

# ─────────────────────────────────────────────
# DB Engine (shared across requests)
# ─────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# ─────────────────────────────────────────────
# OpenRouter / OpenAI client
# ─────────────────────────────────────────────
llm_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ─────────────────────────────────────────────
# Lifespan: startup / shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — verifying DB connection …")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ DB connection OK")
    except Exception as exc:
        logger.error(f"❌ DB connection failed: {exc}")
    yield
    logger.info("🛑 Shutting down — disposing engine …")
    engine.dispose()

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Structured SQL Agent API",
    description="Natural-language → PostgreSQL query agent powered by an LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# API-key auth (optional — skip if not needed)
# ─────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(api_key_header)):
    if APP_API_KEY and key != APP_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key

# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, example="Which department has the highest total salary expense?")

class QueryResponse(BaseModel):
    question:    str
    plan:        dict
    sql:         str
    rows:        list[dict[str, Any]]
    answer:      str

class HealthResponse(BaseModel):
    status: str
    db:     str

# ─────────────────────────────────────────────
# Core agent logic
# ─────────────────────────────────────────────
def get_schema_metadata() -> list[dict]:
    sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]


def generate_query_plan(user_question: str, schema_metadata: list[dict]) -> str:
    schema_text = json.dumps(schema_metadata, indent=2)
    prompt = f"""
You are a PostgreSQL structured query planner.

Schema (JSON):
{schema_text}

User question:
{user_question}

Return ONLY valid JSON (no markdown fences):

{{
  "table": "",
  "select_columns": [],
  "aggregations": [
      {{"function": "", "column": "", "alias": ""}}
  ],
  "filters": [
      {{"column": "", "operator": "=", "value": ""}}
  ],
  "group_by": [],
  "order_by": {{"type": "column|aggregation", "value": ""}},
  "order_direction": "ASC|DESC",
  "limit": 10,
  "explanation": ""
}}

Rules:
- Use EXACT table and column names.
- Do NOT write SQL.
- Do NOT invent columns.
- If no aggregation needed, return empty list.
- If no filters, return empty list.
"""
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def validate_plan(plan_json: str, schema_metadata: list[dict]) -> dict:
    """Parse + validate; raise HTTPException on failure."""
    try:
        # strip accidental markdown fences
        clean = plan_json.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        plan = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"LLM returned invalid JSON: {exc}")

    tables: dict[str, list[str]] = {}
    for row in schema_metadata:
        tables.setdefault(row["table_name"], []).append(row["column_name"])

    table = plan.get("table")
    if table not in tables:
        raise HTTPException(status_code=422, detail=f"LLM chose unknown table: {table!r}")

    valid_cols = set(tables[table])

    for col in plan.get("select_columns", []):
        if col not in valid_cols:
            raise HTTPException(status_code=422, detail=f"Invalid select column: {col!r}")

    for agg in plan.get("aggregations", []):
        if agg["column"] not in valid_cols:
            raise HTTPException(status_code=422, detail=f"Invalid aggregation column: {agg['column']!r}")

    for f in plan.get("filters", []):
        if f["column"] not in valid_cols:
            raise HTTPException(status_code=422, detail=f"Invalid filter column: {f['column']!r}")

    return plan


def build_sql_from_plan(plan: dict) -> str:
    table = f'"{plan["table"]}"'
    select_parts = [f'"{c}"' for c in plan.get("select_columns", [])]

    for agg in plan.get("aggregations", []):
        expr = f'{agg["function"].upper()}("{agg["column"]}")'
        if agg.get("alias"):
            expr += f' AS "{agg["alias"]}"'
        select_parts.append(expr)

    if not select_parts:
        raise HTTPException(status_code=422, detail="Plan has no columns to select")

    sql = f"SELECT {', '.join(select_parts)} FROM {table}"

    if plan.get("filters"):
        clauses = []
        for f in plan["filters"]:
            val = f["value"]
            if isinstance(val, str):
                val = f"'{val}'"
            clauses.append(f'"{f["column"]}" {f["operator"]} {val}')
        sql += " WHERE " + " AND ".join(clauses)

    if plan.get("group_by"):
        sql += " GROUP BY " + ", ".join(f'"{c}"' for c in plan["group_by"])

    ob = plan.get("order_by", {})
    if ob.get("value"):
        direction = plan.get("order_direction", "ASC")
        if ob["type"] == "column":
            sql += f' ORDER BY "{ob["value"]}" {direction}'
        elif ob["type"] == "aggregation":
            for agg in plan.get("aggregations", []):
                if agg.get("alias") == ob["value"]:
                    sql += f' ORDER BY {agg["function"].upper()}("{agg["column"]}") {direction}'
                    break

    if plan.get("limit"):
        sql += f" LIMIT {int(plan['limit'])}"

    return sql


def _serialize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def execute_sql(sql: str) -> list[dict]:
    if not sql.strip().lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT statements are permitted")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            return [{k: _serialize(v) for k, v in row._mapping.items()} for row in result]
    except Exception as exc:
        logger.error(f"SQL execution error: {exc}\nSQL: {sql}")
        raise HTTPException(status_code=500, detail=f"SQL execution error: {exc}")


def reflect(user_question: str, sql_result: list[dict]) -> str:
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"User Question:\n{user_question}\n\n"
                f"SQL Result:\n{json.dumps(sql_result, indent=2)}\n\n"
                "Explain clearly and concisely. If the result is empty, explain why."
            ),
        }],
        temperature=0,
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"
    return {"status": "ok", "db": db_status}

# Add these imports at the top of main.py
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add this route (before the /query route is fine)
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("index.html")
@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def run_query(
    body: QueryRequest,
    _: str = Depends(require_api_key),
):
    logger.info(f"Question: {body.question!r}")

    schema   = get_schema_metadata()
    plan_raw = generate_query_plan(body.question, schema)
    logger.info(f"Raw plan: {plan_raw}")

    plan = validate_plan(plan_raw, schema)
    sql  = build_sql_from_plan(plan)
    logger.info(f"SQL: {sql}")

    rows   = execute_sql(sql)
    answer = reflect(body.question, rows)

    return QueryResponse(
        question=body.question,
        plan=plan,
        sql=sql,
        rows=rows,
        answer=answer,
    )


@app.get("/schema", tags=["Agent"])
def get_schema(_: str = Depends(require_api_key)):
    """Return the database schema (table → columns)."""
    metadata = get_schema_metadata()
    grouped: dict[str, list[str]] = {}
    for row in metadata:
        grouped.setdefault(row["table_name"], []).append(row["column_name"])
    return grouped

# ─────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
