# Deploying to Streamlit Community Cloud

The repository is already structured the way Streamlit Community Cloud expects:

| Requirement | Status |
|---|---|
| Public GitHub repository | ✅ `tausnix/telecom-churn-intelligence` |
| Entrypoint file in the repo root | ✅ `app.py` |
| Dependency file in the repo root | ✅ `requirements.txt` |
| Model artefacts committed | ✅ `models/*.joblib` + `models/model_metadata.json` |
| Figures available to the app | ✅ `reports/figures/` |
| Secrets required | ✅ none |

Because the trained models and figures are committed, the deployed app has nothing to train
and starts in seconds.

## One-click deploy

The badge in the README uses a pre-filled deploy link:

```text
https://share.streamlit.io/deploy?repository=tausnix/telecom-churn-intelligence&branch=main&mainModule=app.py
```

Opening it while signed in skips the repository/branch/file form and takes you straight to
the confirmation screen.

> **Subdomain note:** the obvious `telecom-churn-intelligence` subdomain is already taken by
> an unrelated app, so Streamlit Cloud will reject it. `tausnix-churn-intelligence` is
> verified free and is the recommended choice below.


## Manual steps

1. Open <https://share.streamlit.io> and click **Continue to sign-in** → **GitHub**.
   Authorise Streamlit (read access to public repositories is enough).
2. Click **Create app** → **Yup, I have an app**.
3. Fill in:

   | Field | Value |
   |---|---|
   | Repository | `tausnix/telecom-churn-intelligence` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL (Advanced) | `tausnix-churn-intelligence` |

4. Open **Advanced settings** → set **Python version** to **3.12** (the Cloud default).
   Leave *Secrets* empty — the app needs none.
5. Click **Deploy**. The first build takes ~2–4 minutes (installing pandas, scikit-learn,
   matplotlib, seaborn and Streamlit).

Resulting URL:

```text
https://tausnix-churn-intelligence.streamlit.app
```

## Verify after deploy

- The hero strip shows **7,043 customers · 3 tuned models · best Random Forest**.
- **Score a customer** → *Predict churn risk* returns a probability, risk band and drivers.
- **Model performance** → six metric cards plus the comparison table.
- **Data insights** → EDA gallery and dataset download.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **`Error running app.`** with `👋 Welcome to Streamlit!` → `Email:` in the logs, then `connection refused` on port 8501 | `headless` was `false` in `.streamlit/config.toml`. Without a credentials file (Cloud has none) Streamlit blocks on an interactive email prompt and never binds its port. Keep `headless = true` — it is the committed default. |
| `ModuleNotFoundError: No module named 'src'` | Main file path must be `app.py` at the repo root, not a file inside `src/`. |
| `InconsistentVersionWarning` when loading a model | Dependencies drifted from the training environment. `requirements.txt` pins the exact versions (`pandas`, `numpy`, `scikit-learn`, `streamlit`, `matplotlib`, `seaborn`); rebuild the app after changing dependencies. |
| Build log shows `Detected pyarrow … Replacing with pyarrow<25` | Streamlit Cloud force-replaces `pyarrow>=25` (known segfault, apache/arrow#50471). `requirements.txt` already pins `pyarrow>=24,<25` so the replacement happens during resolution instead of after install. |
| Build fails installing a package | Open **Manage app → Logs** for the pip error, then adjust `requirements.txt`. |
| App sleeps after inactivity | Expected on the free tier. Opening the URL wakes it (~10–30 s). |
| Charts missing | `reports/figures/` must be committed — they are, but a `.gitignore` mis-edit could drop them. |

## Alternative: keep it local

```bash
python -m streamlit run app.py
```

Nothing in the code is Cloud-specific; the same command works on any machine with the
dependencies installed.
