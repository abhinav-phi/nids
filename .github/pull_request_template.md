<!-- Provide a general summary of the change in the Title above -->

## Description

<!-- Describe what this PR changes and why. Reference the notebook(s)/docs affected. -->

## Motivation & Context

<!-- Why is this change required? Link related issues with "Fixes #N" / "Closes #N". -->

## Type of Change

<!-- Mark the relevant option with an "x" — exactly one primary type. -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing notebooks/artifacts to fail)
- [ ] 📓 Notebook change (training / inference / EDA / dashboard logic)
- [ ] 📝 Documentation update
- [ ] 🔧 Build / CI / tooling
- [ ] ♻️ Refactor (no behavior change)

## Areas Touched

<!-- Mark all that apply. -->

- [ ] `_build/01_EDA.py` (EDA notebook)
- [ ] `_build/02_Training_GPU.py` (training pipeline)
- [ ] `_build/03_Inference_API.py` / `frag_inference.py` / `frag_api.py` (inference + API)
- [ ] `_build/04_Dashboard.py` (Gradio dashboard)
- [ ] `docs/` (specification set)
- [ ] `README.md` / `.github/`

## Checklist

<!-- The first two are enforced by CI — local run before pushing. -->

- [ ] `python _build/pack.py` runs clean (notebooks regenerate, nbformat + syntax validation pass)
- [ ] `python _build/smoke_test.py` prints **SMOKE TEST PASSED**
- [ ] Notebook logic changes were executed top-to-bottom on Colab (T4) without errors
- [ ] Artifact contract preserved: `model.pkl` sklearn-compatible, `n_features_in_ == 52`, StandardScaler
- [ ] Determinism: all new randomness uses `random_state=42` / seeded generators
- [ ] GPU cells degrade gracefully to CPU
- [ ] No secrets, no dataset files, no generated artifacts committed
- [ ] Docs updated (`docs/NIDS_*.md` / README) if behavior or paths changed
- [ ] Conventional commit title (`feat:`, `fix:`, `docs:`, `chore:`, …)

## Screenshots / Output

<!-- For notebook cells or dashboard changes, paste key outputs (comparison table, ROC/AUC, summary) here. -->
