# Support

Thanks for using **The Sentinel NIDS**! Here's how to get help quickly.

---

## 📖 Read the docs first (most questions are answered here)

| Question | Where |
|----------|-------|
| How do I run the project on Colab? | [`nids-backend/notebooks/colab/README.md`](nids-backend/notebooks/colab/README.md) |
| What does each notebook do? | [README — Notebooks table](README.md) |
| Dataset setup (where to put the CSV) | Colab README → *One-time setup* |
| How artifacts flow between notebooks | Colab README → *How artifacts flow* |
| Full product requirements / architecture | [`docs/NIDS_PRD.md`](docs/NIDS_PRD.md) · [`docs/NIDS_TechSpec.md`](docs/NIDS_TechSpec.md) |
| What deliberately doesn't run on Colab | Colab README → *What deliberately does NOT run on Colab* |
| Contributing / build system | [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) |

---

## 🆘 Quick troubleshooting

| Symptom | Fix |
|---------|-----|
| `Dataset not found!` assertion in notebook 01/02 | Upload `cicids2017_cleaned.csv` to **Google Drive → `MyDrive/nids_data/`**, then re-run |
| `Trained artifacts not found!` in notebook 03/04 | Run **`02_Training_GPU_Colab.ipynb`** first (it syncs to `MyDrive/nids_artifacts/`), or upload `nids_artifacts.zip` to `/content/` |
| `NO GPU — go to Runtime → Change runtime type` | Select **T4 GPU** runtime and re-run the cell |
| Optuna / SHAP cell fails on NumPy | Colab handles versions automatically; locally install `numpy<2` (see CONTRIBUTING) |
| Dashboard shows "Backend offline" | Re-run the API/server cells of notebook 04 (or 03) |

---

## 🗣️ Ask a question

- **[GitHub Discussions](https://github.com/abhinav-phi/nids/discussions)** — best place for usage questions, ideas and sharing your results.

## 🐛 Found a bug?

Open an issue with the **[Bug report template](https://github.com/abhinav-phi/nids/issues/new?template=1-bug-report.md)** — include the notebook, cell number, runtime, and the full error output.

## 🔒 Security issue?

**Never** in a public issue — use [private vulnerability reporting](https://github.com/abhinav-phi/nids/security/advisories/new) (see [SECURITY.md](.github/SECURITY.md)).

## ⏱️ Response expectations

This is a solo-maintained project — expect responses within a few days. CI runs on every push/PR, so `pack.py` + `smoke_test.py` failures are caught automatically.
