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



**Primary UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/) · **Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) · **Health:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

If the UI is on another port (e.g. 8001), replace `8000` in the links.

### Conda `base` vs venv

Installing into **Anaconda base** can upgrade NumPy/sklearn and **break** other packages (numba, conda tools). Prefer a **project venv** as above. After any sklearn upgrade, run **`python3 scripts/train_model.py` again** to clear pickle version warnings.

### `zsh: command not found: #`

Do not paste comment lines that start with `#` as their own command; only run the actual shell commands.

**Streamlit** (second terminal):

```bash
cd tractor-predictive-maintenance/dashboard
streamlit run streamlit_app.py
```

Set **TRACTOR_API_URL** if the API is not on port 8000.

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

## GitHub and Vercel (live UI)

**Suggested repository name** (GitHub does not allow spaces): `ai-powered-tractor-predictive-maintenance`.

```bash
cd tractor-predictive-maintenance
git init
git add -A
git commit -m "Initial commit: tractor predictive maintenance demo"
gh repo create ai-powered-tractor-predictive-maintenance --public --source=. --remote=origin --push
```

If you do not use GitHub CLI, create an empty repo with that name on GitHub, then `git remote add origin https://github.com/<you>/ai-powered-tractor-predictive-maintenance.git` and `git push -u origin main`.

**Vercel** hosts the **static** site in `frontend/` well; the **FastAPI** API (models, `/predict`, `/docs`) should run on a Python host (e.g. [Render](https://render.com), Railway, Fly.io) or `docker compose` on a VPS.

1. Deploy the API and copy its public origin (no trailing slash), e.g. `https://tractor-api-xxxx.onrender.com`.
2. In [Vercel](https://vercel.com) → Import the GitHub repo → leave **Root Directory** as the repo root (this project’s `vercel.json` sets `outputDirectory` to `frontend` and runs `node scripts/inject-api-base.mjs`).
3. Under **Environment Variables**, add **`TRACTOR_API_BASE`** = that API origin. Redeploy so the console’s default API URL points at your live backend.
4. On the API host, set **CORS** so your `*.vercel.app` origin is allowed (see `backend/app.py` / env docs if present).

Without `TRACTOR_API_BASE`, the hosted UI still defaults to `http://127.0.0.1:8000`; users can paste the real API URL in the console field.

**Fastest live link from your machine** (Cursor’s cloud agents cannot call Vercel from here; you run this once locally):

```bash
cd tractor-predictive-maintenance
bash scripts/deploy-vercel.sh
```

The CLI prints a `https://….vercel.app` preview URL. Use `bash scripts/deploy-vercel.sh --prod` after `vercel link` for a stable production URL. Set `TRACTOR_API_BASE` first if your API is already hosted.

**Fully automated after GitHub push:** add repository secrets `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`, and optionally `TRACTOR_API_BASE`. Pushes to `main` run `.github/workflows/deploy-vercel.yml`.

## Resume headline

**AI-Powered Tractor Predictive Maintenance and Fault Diagnosis System** — Python, scikit-learn, XGBoost (optional), SMOTE, FastAPI, Streamlit, SQLAlchemy, LangChain/OpenAI, cited RAG.
