# Deploying to Streamlit Community Cloud (free)

Time needed: about 10–15 minutes. You need a free GitHub account and the project folder from this package.

---

## Step 1: Create the GitHub repository

1. Sign in at <https://github.com> and click **New repository** (the **+** in the top-right).
2. Name it, for example, `hypertension-risk-screening`.
3. Choose **Public**. Community Cloud's free tier deploys public repositories; private ones are possible but
   limited.
4. Leave "Add a README" **unticked**, because the project already has one. Click **Create repository**.

## Step 2: Upload the project

**Option A: in the browser (no Git needed)**

1. On the new, empty repository page, click **uploading an existing file**.
2. Open the unzipped `htn-screening-app` folder on your computer, select **everything inside it**, and drag it
   into the browser window. Drag the contents, not the folder itself, so that `app.py` sits at the top level of
   the repository.
3. **Check that the hidden items came across:** `.streamlit/config.toml`, `.gitignore` and `.python-version`.
   Windows and macOS hide files that start with a dot:
   * Windows File Explorer: **View → Show → Hidden items**.
   * macOS Finder: press **Cmd + Shift + .**
   If `.streamlit/config.toml` is missing after upload, click **Add file → Create new file**, type
   `.streamlit/config.toml` as the name and paste in the file's contents.
4. Click **Commit changes**.

**Option B: with Git**

```bash
cd htn-screening-app
git init
git add .
git commit -m "Hypertension Risk Screening System"
git branch -M main
git remote add origin https://github.com/<your-username>/hypertension-risk-screening.git
git push -u origin main
```

**Check before going public:** the repository must contain **no** `nhanes_cohort*.csv`, `checkpoints/`,
`logs/`, the project PDF or the results ZIP. `.gitignore` blocks these for Option B; with Option A, simply do not
add them.

## Step 3: Deploy

1. Go to <https://share.streamlit.io> and **Continue with GitHub**. Authorise Streamlit to read your repositories.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   * **Repository:** `<your-username>/hypertension-risk-screening`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL:** choose a subdomain, e.g. `htn-screening-miva` → `https://htn-screening-miva.streamlit.app`
4. Click **Advanced settings** and set **Python version = 3.13**. ⚠️ **This step is essential.**
   Community Cloud ignores `.python-version` and may default to a newer Python, for which
   scikit-learn 1.6.1 (the version the model was saved with) has no installable package. No secrets are needed.
   Click **Save**.
5. Click **Deploy**. The first build takes about 3–6 minutes while it installs SHAP, scikit-learn and the other
   libraries. When it finishes, the app opens at your URL.

## Step 4: Check the live app

1. Home loads with the navy header and the disclaimer.
2. **Screen for Risk → Demonstration mode → Profile A → Run model** shows **Moderate Risk** with a SHAP chart.
3. Profiles B and C show **Higher Risk**.
4. **Research Analytics** shows all seven tabs with charts.

If all four work, the deployment matches the tested build.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build log says `No matching distribution found for scikit-learn==1.6.1` or shows Python 3.14 | The Python version was not set to 3.13. In the app's **⋮ → Settings** you cannot change Python after deployment: **delete the app and redeploy**, choosing 3.13 under Advanced settings. |
| `ModuleNotFoundError: htn` or `views` | `app.py` is not at the repository root. The folders `htn/`, `views/`, `model/` and `research_results/` must sit beside it. |
| `Model bundle not found` | `model/model_bundle.joblib` was not uploaded (≈ 300 KB). |
| Theme looks like plain Streamlit | `.streamlit/config.toml` is missing (see Step 2, point 3). The custom CSS still applies. |
| App "falls asleep" | Free apps sleep after a period without visitors. Open the URL and click **Yes, get this app back up**; it wakes in under a minute. Open it a few minutes before a demonstration. |

## Updating the app

Edit files on GitHub (or `git push`). Community Cloud redeploys automatically within about a minute.
