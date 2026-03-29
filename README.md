# AI-Powered Tractor Predictive Maintenance and Fault Diagnosis

End-to-end demo that classifies **Healthy / Warning / Critical** bands, predicts **failure type** (HDF, PWF, OSF, TWF, RNF mapped to readable labels), optionally calls an **LLM** with **RAG-style** excerpts from `data/tractor_fault_manual.csv`, logs requests to **SQLite or PostgreSQL**, and ships a **web UI** (full-bleed landing + operations console) from the API root, plus an optional **Streamlit** dashboard.

Industrial framing uses the public **AI4I 2020** predictive maintenance schema (air/process temperature, RPM, torque, tool wear).

## Architecture

- **Data**: `data/ai4i2020.csv` (Kaggle AI4I or synthetic bootstrap). Features are engineered in `backend/feature_engineering.py` (shared by training and API).
- **Mechanical power**: \(P_{\mathrm{kW}} = T_{\mathrm{Nm}} \cdot n_{\mathrm{rpm}} / 9549\). **PWF** rule-of-thumb band is **3.5–9 kW** (equivalent to the original 3500–9000 W specification). The legacy bug of treating that value as “watts” in the UI is fixed.
- **Training**: `scripts/train_model.py` — SMOTE on the training split, **RandomForest** with `class_weight='balanced_subsample'`, optional **XGBoost** when OpenMP is available. Model selection uses **macro-F1** (not accuracy alone). Outputs **confusion matrix**, **per-class metrics**, and **stratified CV macro-F1** on the imbalanced training fold (see `models/metrics.json`).
- **Inference**: `backend/app.py` (FastAPI) loads `backend/model.pkl`; if unpickling fails (e.g. XGBoost without `libomp`), it **falls back** to `models/random_forest.pkl`.
- **UI**: Static assets under `frontend/`; optional Streamlit in `dashboard/`.
- **Diagnosis**: `backend/llm_helper.py` returns **structured manual citations** plus optional OpenAI text.

## Business impact

- Reduces **unplanned downtime** by surfacing failure class and urgency before a breakdown.
- Gives **field-actionable** guidance tied to maintenance snippets (citations), improving trust vs. a black-box score alone.
- **Logs** support audit and post-incident review; API shape (`/model_info`, paginated `/logs`) is suitable for integration into OEM or fleet tooling (demo scope).

## Known limitations

- **Class imbalance**: “Healthy” dominates; **accuracy can look strong while macro-F1 and rare-class recall are weaker** — always read `metrics.json` (confusion matrix + CV).
- **Probabilities** are raw `predict_proba` outputs, **not isotonic-calibrated** unless you add calibration.
- **Single estimator** is deployed (best of RF vs XGB), not a stacked ensemble.
- **RNF** remains inherently hard; SMOTE + class weights help but do not guarantee detection.
- **Rule hints** in the API are **heuristic** (AI4I-style), not SHAP.

## Local development

```bash
cd tractor-predictive-maintenance
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python3 scripts/train_model.py
cd backend && python3 -m uvicorn app:app --reload --port 8000
```

**Local UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/) · **Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) · **Health:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

If the UI is on another port (e.g. 8001), replace `8000` in the links.

## Live deployment

**Live UI:** [https://ai-based-tractor-predictive-maintenance.onrender.com/](https://ai-based-tractor-predictive-maintenance.onrender.com/) · **Docs:** [https://ai-based-tractor-predictive-maintenance.onrender.com/docs](https://ai-based-tractor-predictive-maintenance.onrender.com/docs) · **Health:** [https://ai-based-tractor-predictive-maintenance.onrender.com/health](https://ai-based-tractor-predictive-maintenance.onrender.com/health)

Frontend: https://ai-based-tractor-predictive-maintenance-57yef6dwt.vercel.app/
API Docs: https://ai-based-tractor-predictive-maintenance.onrender.com/docs
Backend: https://ai-based-tractor-predictive-maintenance.onrender.com/

- Primary production hosting is **Render** using `Dockerfile.api`.
- The hosted frontend and API run on the **same origin** (`onrender.com`), so no separate frontend host is required.
- The console now defaults to the **current site origin** in hosted environments and ignores stale saved `localhost` API base values from earlier local testing.

### Conda `base` vs venv

Installing into **Anaconda base** can upgrade NumPy/sklearn and **break** other packages (numba, conda tools). Prefer a **project venv** as above. After any sklearn upgrade, run **`python3 scripts/train_model.py` again** to clear pickle version warnings.



### macOS / XGBoost

If XGBoost fails to load (`libomp`), install OpenMP (`brew install libomp`) and retrain, or rely on the **Random Forest** fallback at runtime.

### Retrain after feature changes

If you change `FEATURE_LIST` (e.g. `power` → `power_kw`), run **`python3 scripts/train_model.py`** again so `backend/feature_list.pkl` matches the code.

## Official dataset (recommended)

1. Download [Predictive Maintenance Dataset AI4I 2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).
2. Save as `data/ai4i2020.csv` (or `python3 scripts/download_kaggle_dataset.py`).
3. Re-run `python3 scripts/train_model.py`.

## API (selected)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Classify + `explanation_factors`, `derived.power_kw` |
| POST | `/predict_with_diagnosis` | Above + `llm_diagnosis` + `manual_citations` |
| GET | `/logs` | `?limit=&offset=&prediction_contains=&health_status=` |
| GET | `/model_info` | Feature names, PWF band, loaded artifact |
| GET | `/health` | Liveness |

## Tests & CI

```bash
python3 scripts/train_model.py
pytest tests/ -q
```

GitHub Actions (`.github/workflows/ci.yml`): install deps, train, pytest.

## Docker

```bash
docker compose build api
docker compose up api
```

`Dockerfile.api` trains on build if artifacts are missing and serves the API + static UI on port 8000.

## LLM diagnosis

- Set `OPENAI_API_KEY` in `.env` (see `.env.example`) for GPT-backed text; citations are still returned for traceability.

## PostgreSQL

`DATABASE_URL=postgresql+psycopg2://...` — see `docker-compose.yml` for a local Postgres service.

## GitHub and Render

**Suggested repository name** (GitHub does not allow spaces): `ai-powered-tractor-predictive-maintenance`.

```bash
cd tractor-predictive-maintenance
git init
git add -A
git commit -m "Initial commit: tractor predictive maintenance demo"
gh repo create ai-powered-tractor-predictive-maintenance --public --source=. --remote=origin --push
```

If you do not use GitHub CLI, create an empty repo with that name on GitHub, then `git remote add origin https://github.com/<you>/ai-powered-tractor-predictive-maintenance.git` and `git push -u origin main`.

**Render** is the primary recommended deployment target for this repo because it runs the FastAPI API and serves the static frontend from the same origin.

1. Push the repo to GitHub.
2. In Render, create a **Web Service** from the GitHub repo with:
   - **Runtime** = `Docker`
   - **Dockerfile Path** = `Dockerfile.api`
   - **Health Check Path** = `/health`
3. Add environment variables:
   - `CORS_ORIGINS=*`
   - `OPENAI_API_KEY` and `OPENAI_MODEL` if you want GPT-backed diagnosis
   - optional `DATABASE_URL` if you want persistent PostgreSQL logs
4. If you do **not** set `DATABASE_URL`, the app falls back to SQLite for a quick demo deploy.
5. If you want persistent logs, create a Render Postgres database and paste its **Internal Database URL** into `DATABASE_URL`, then redeploy.

`render.yaml` is included for Blueprint-based deployment, but the current live service also works with the dashboard’s standard Git-backed Docker flow.

## Optional GitHub and Vercel frontend deploy

If you want to keep the GitHub Actions workflow at [deploy-vercel.yml](/Users/adarshgupta/tractor-predictive-maintenance/.github/workflows/deploy-vercel.yml), add these **repository secrets** in GitHub:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`
- `TRACTOR_API_BASE`

Recommended values:

- `TRACTOR_API_BASE=https://ai-based-tractor-predictive-maintenance.onrender.com`
- `VERCEL_TOKEN` = token from Vercel account settings
- `VERCEL_ORG_ID` = your Vercel team ID (for a personal hobby account, use your default team ID)
- `VERCEL_PROJECT_ID` = the Vercel project ID for this frontend

High-level setup:

1. Create or import this repo as a Vercel project.
2. In Vercel, copy the **Team ID** and **Project ID** from project/account settings.
3. In Vercel, create a personal access token.
4. In GitHub, open **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret** and add the four secrets above.
5. Push to `main` again, or re-run the failed workflow from the GitHub Actions page.

The frontend now loads `/assets/js/api-config.js`, so the GitHub Action can inject the Render backend URL into the Vercel-hosted UI before deployment.

## Resume headline

**AI-Powered Tractor Predictive Maintenance and Fault Diagnosis System** — Python, scikit-learn, XGBoost (optional), SMOTE, FastAPI, Streamlit, SQLAlchemy, LangChain/OpenAI, cited RAG.
