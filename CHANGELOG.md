# Changelog

All notable changes to **The Sentinel — Network Intrusion Detection System** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- NSL-KDD / UNSW-NB15 multi-dataset generalization experiments
- Live topology graph of aggregated flows (Gradio `graphviz` panel)
- Model registry + automated retraining experiments (MLflow on Colab)

---

## [1.0.0] — 2026-08-28 · Colab Edition

The complete rewrite of the deployment model: the entire NIDS — EDA, T4-GPU training, inference, embedded API, and dashboard — now runs inside **4 Google Colab notebooks**, generated from versioned Python sources.

### Added

**Notebook suite** (all T4-GPU aware, artifact contract preserved)
- **`01_EDA_Colab.ipynb`** — CICIDS2017 exploration: class distribution & imbalance, data-quality audit, feature distributions, correlation analysis, attack-vs-normal behaviour, key-numbers summary table.
- **`02_Training_GPU_Colab.ipynb`** — the full training pipeline: chunked stratified sampling (~400k rows, float32), cleaning, label encoding, stratified split, dual scalers (Standard + Robust), **7-model arena** ranked by Macro F1, **XGBoost on T4 GPU** with a CPU-vs-GPU timing bake-off, **PyTorch MLP (256-128-64)** with class-weighted loss and early stopping, **Optuna Bayesian tuning** (20 trials), **per-class ROC curves + AUC**, **normalized confusion matrix**, per-class F1 + benign FPR, SHAP smoke test, GPU benchmark table, and a training-summary table.
- **`03_Inference_API_Colab.ipynb`** — the embedded backend: production inference stack (severity map, cached SHAP `TreeExplainer`, artifact parity guards), the 52-feature `FlowExtractor`, synthetic attack flows (DDoS / port-scan / brute force / normal), **global + per-class SHAP deep-dive plots**, a **FastAPI server** (`/health`, `/api/predict`, `/api/alerts`, `/api/stats`, `/api/ip-leaderboard`, `/ws/live`) with the full validation-guard contract, balanced **CSV replay**, an **API contract summary**, a public **Cloudflare tunnel**, and an optional **Gemini mini-chatbot**.
- **`04_Dashboard_Colab.ipynb`** — a **Gradio command center** (KPI strip, attack pie, severity bars, 3-second live alert feed, attacker leaderboard, attack timeline, per-alert SHAP explainer, traffic-injection lab) plus static dashboard PNGs for reports.

**Build & verification system** (`nids-backend/notebooks/colab/_build/`)
- Percent-format notebook sources (`01_EDA.py`, `02_Training_GPU.py`, `03_Inference_API.py`, `04_Dashboard.py`) with shared fragments (`frag_discovery`, `frag_inference`, `frag_api`, `frag_server`).
- `pack.py` — regenerates all four `.ipynb` files with T4-GPU metadata; runs syntax + nbformat validation on every rebuild.
- `smoke_test.py` — executes notebook 03's embedded inference + API stack end-to-end and asserts the production contract (predictions, persistence, every 400/422 validation guard, severity mapping). **Status: PASSED.**

**Project hygiene**
- `README.md` rewritten for the Colab-only workflow; `docs/` specification set (PRD, TechSpec, AppFlow, Design, Schema, ImplementationPlan, Tracker, Rules) fully synced.
- `.github/` community kit: `SECURITY.md` (private disclosure policy), `CONTRIBUTING.md` (build-system workflow), `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `CODEOWNERS`, `FUNDING.yml`, PR template, structured issue templates, and a **GitHub Actions CI** workflow (`pack.py` + `smoke_test.py` on every push/PR).

### Changed

- **ISSUE-05 (fixed)** — notebook label detection now resolves the CSV's `Attack Type` column directly (previously printed `Label column: None`).
- **ISSUE-06 (fixed)** — benign FPR is computed with the benign index looked up from the label encoder (the old `evaluate.py` hardcoded class 0, which was `Bots`).
- Model suite refocused: Optuna-tuned XGBoost and a PyTorch MLP replace the SVM-RBF / Voting / Stacking entries of the local 9-model suite (ranking metric unchanged: Macro F1).
- Documentation legend extended with 🟠 LOCAL-ONLY for components present only in git history.

### Removed

- **Local-only runtime** (kept in git history): `nids-backend/src/` (FastAPI backend, sniffer, extractor, simulators), `nids-frontend/` (React dashboard), `tests/` (78 pytest cases), `venv/` (~1.1 GB), `model.pkl`/scalers/`nids.db`, and the three original notebooks — all superseded by the notebook-embedded implementations and the `_build/` smoke test.
- Root utility dumps and scripts (`combine_files.py`, `combine_nids_clean.py`, `remove_comments.py`, codebase txt/md), historical `NIDS_AuditReport.md`, `NIDS_Final_Report.pdf`, `VIVA_GUIDE.md` — ~1.4 GB freed overall.

---

## [0.x] — Local Edition (pre-2026-08-28)

The original local-machine implementation, preserved in git history (`main` before `a577e28`):

- FastAPI backend (`src/api/`) with strict input validation, optional API-key auth, per-IP rate limiting, WebSocket alert streaming, SQLite/WAL persistence, and a hardened LangChain + Gemini chatbot.
- Scapy packet capture (`src/capture/sniffer.py`) with 5-tuple flow assembly, retry/backoff delivery, and counters.
- 9-model training pipeline (`src/model/train.py`), 52-feature `FlowExtractor` (`src/features/extractor.py`), SHAP-explained inference (`src/model/predict.py`).
- React 18 + TypeScript + Vite command-center dashboard with vitest suites (21 specs); 78 pytest backend tests.
- Final security-hardening pass (PH10): WS auth + client cap, body-size cap, metadata IP validation, event-loop offloading, timezone-aware UTC, SHAP JSON safety, corrupt-artifact degradation (113 tracked items | 104 done).

---

[unreleased]: https://github.com/abhinav-phi/nids/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/abhinav-phi/nids/commits/main
