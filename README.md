# APEX-PULSE

**24/7 Cloud-Ready Multi-Market (Bybit Crypto + Forex) Algorithmic Trading Bot — Phase 1 Foundation**

APEX-PULSE provides a cross-platform (Linux / Windows 11), zero-C-extension FastAPI + SQLAlchemy 2.x trading engine with a mobile-first obsidian dark-mode monitoring dashboard.

## Safety & Architecture Highlights

- **Hard-Locked `PAPER` Mode:** Defaults to `TRADING_MODE=PAPER` across `.env.example` and `app/config.py`.
- **Database Flexibility:** Uses SQLite with `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON` by default; automatically switches to PostgreSQL when `DATABASE_URL` is configured.
- **Emergency Kill Switch:** Immediately halts the trading loop, closes all open paper positions, persists state in DB, and logs a `CRITICAL` audit event.
- **Modular Structure:**
  - `app/config.py` — Typed `pydantic-settings` configuration
  - `app/database.py` — SQLAlchemy 2.x engine, session factory, and WAL initialization
  - `app/models.py` — `trades`, `equity_snapshots`, `bot_state`, `strategy_configs`, `system_logs`
  - `app/brokers/base.py` & `app/brokers/paper_sim.py` — Abstract `BaseBroker` + realistic `PaperSimBroker`
  - `app/engine/runner.py` — Lifespan-managed async `BotRunner`
  - `app/main.py` & `app/templates/index.html` — FastAPI REST API + mobile-first dashboard

## Quick Start (Windows 11 PowerShell)

```powershell
git clone https://github.com/peacefulme232-debug/apex-pulse-bot.git; cd apex-pulse-bot
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in your browser.

## Running Tests

```powershell
pytest -v
```