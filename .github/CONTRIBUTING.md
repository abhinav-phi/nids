# Contributing to The Sentinel NIDS

First off, thank you for considering a contribution! 🛡️

This project is a Colab-only, ML-powered Network Intrusion Detection System. The entire implementation — EDA, T4-GPU training, inference, embedded FastAPI backend, and the Gradio dashboard — lives inside **4 Colab notebooks** that are **generated** from readable Python sources. Contributions follow that build system.

---

## The Golden Rule

> **Never edit the `.ipynb` files directly.** They are build artifacts.
>
> Edit the percent-format sources in `nids-backend/notebooks/colab/_build/`, then rebuild.

```
_edit_ _build/02_Training_GPU.py  →  python _build/pack.py  →  notebooks regenerated
```

This keeps diffs reviewable, keeps the four notebooks in sync with shared fragments (`frag_*.py`), and lets CI validate everything.

---

## Development Environment

You only need local Python for **building and testing** — the notebooks themselves run on Google Colab.

| Requirement | Purpose |
|-------------|---------|
| Python 3.10+ | Running `pack.py` and `smoke_test.py` |
| `pip install "numpy<2" pandas scikit-learn shap fastapi "uvicorn[standard]" sqlalchemy requests joblib matplotlib nbformat` | Local validation toolchain |
| Google Drive | Only for actually running the notebooks (dataset + artifacts) |

> ⚠️ `numpy<2` matters locally: older `shap` builds break on NumPy 2.x. Colab resolves this automatically via its own pinned stack.

---

## Contribution Workflow

1. **Fork & branch** — `feat/<short-name>`, `fix/<short-name>`, or `docs/<short-name>`.
2. **Make your change** in `nids-backend/notebooks/colab/_build/*.py` (and `docs/` if behavior or paths changed).
3. **Rebuild the notebooks:**
   ```bash
   cd nids-backend/notebooks/colab/_build
   python pack.py        # regenerates all 4 .ipynb files + nbformat/syntax validation
   ```
4. **Run the smoke test:**
   ```bash
   python smoke_test.py  # must print: SMOKE TEST PASSED
   ```
   This executes notebook 03's embedded inference + API stack and asserts the full production contract (predictions, persistence, every 400/422 validation guard, severity mapping).
5. **(Notebook-logic changes)** Run the affected notebook top-to-bottom on Colab with the T4 runtime and confirm the summary tables.
6. **Open a Pull Request** using the provided template.

---

## What Changes Where

| Area | File(s) |
|------|---------|
| EDA notebook | `_build/01_EDA.py` |
| Training pipeline (models, Optuna, ROC/AUC) | `_build/02_Training_GPU.py` |
| Inference, `FlowExtractor`, FastAPI, replay | `_build/03_Inference_API.py` + `frag_*.py` |
| Gradio dashboard | `_build/04_Dashboard.py` |
| Shared inference stack | `_build/frag_inference.py` (single source of the 52-feature contract & severity map) |
| Shared API app | `_build/frag_api.py` |
| Documentation | `docs/NIDS_*.md`, `README.md`, `nids-backend/notebooks/colab/README.md` |

---

## Code Standards

- **Determinism:** every random operation uses `random_state=42` / seeded generators.
- **Artifact contract is sacred:** `model.pkl` must stay a sklearn-compatible estimator with `n_features_in_ == 52`; `scaler.pkl` stays a StandardScaler. Notebooks 03/04 depend on it.
- **No silent coercion:** malformed feature vectors are rejected (422), never "fixed" — mirror the validation semantics in `frag_api.py`.
- **Graceful GPU degradation:** CUDA cells must fall back to CPU with a clear warning, never crash.
- **No secrets:** API keys are entered via `getpass` per session. Never commit keys, tokens, or dataset files.
- **Style:** PEP 8, type hints on function signatures, docstrings for public helpers, section headers in notebook sources (`# %% [markdown]`).

---

## Commit & PR Conventions

- **Conventional commits:** `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:` — e.g. `feat(colab): add per-class ROC curves to training notebook`.
- **One logical change per PR.** The repo history is split into reviewable units on purpose.
- **Docs stay in sync:** any behavior/path change gets a matching update in the relevant `docs/NIDS_*.md` and the Tracker (see `docs/NIDS_Rules.md`).

---

## Reporting Bugs & Suggesting Features

- Bugs → [Issue: Bug report](https://github.com/abhinav-phi/nids/issues/new?template=1-bug-report.md)
- Features → [Issue: Feature request](https://github.com/abhinav-phi/nids/issues/new?template=4-feature-request.md)
- Questions → [GitHub Discussions](https://github.com/abhinav-phi/nids/discussions)
- **Security vulnerabilities → [Private disclosure](https://github.com/abhinav-phi/nids/security/advisories/new) — never a public issue** (see [SECURITY.md](SECURITY.md))

---

## Licensing

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE) that covers the project.
