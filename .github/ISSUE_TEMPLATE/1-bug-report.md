---
name: 🐛 Bug report
about: Something in a notebook, the embedded API, or the build system is broken
title: "[Bug]: "
labels: ["bug", "triage"]
assignees: ["abhinav-phi"]
---

## Affected Component

<!-- Which notebook / part of the project is misbehaving? -->

- [ ] `01_EDA_Colab.ipynb` — EDA
- [ ] `02_Training_GPU_Colab.ipynb` — training / Optuna / evaluation
- [ ] `03_Inference_API_Colab.ipynb` — inference, FlowExtractor, FastAPI
- [ ] `04_Dashboard_Colab.ipynb` — Gradio dashboard
- [ ] `_build/` — build system (`pack.py`, `smoke_test.py`)
- [ ] `docs/` — documentation mismatch

## Environment

- **Colab runtime:** <!-- T4 GPU / CPU / other -->
- **Notebook cell number / step:** <!-- e.g. 02, Step 8.5 (Optuna) -->
- **Artifact state:** <!-- does /content/nids_artifacts/ exist? which cells ran before? -->
- **Dataset location:** <!-- MyDrive/nids_data/ or /content/ -->

## What Happened

<!-- A clear, concise description of the bug. -->

## Steps to Reproduce

1.
2.
3.

**Expected behavior:**

**Actual behavior / error output:**

```
<!-- Paste the full traceback or cell output here -->
```

## Additional Context

<!-- Screenshots, plots, or anything else that helps. -->

> ⚠️ **Security vulnerabilities are reported privately** — see [SECURITY.md](../SECURITY.md), not this template.
