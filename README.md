# Food Policy Media Sentiment Analysis Dashboard

An interactive [Shiny for Python](https://shiny.posit.co/py/) app for exploring the relationship between **media coverage** (volume and sentiment) and **food policy actions** in the U.S.  
Built as part of **FIN-6305 Quantitative Methods III**, University of Denver.

---

## Features

- 📈 **Model performance overview**: Random Forest and LightGBM cross-validated ROC/PR metrics
- 🔎 **Prediction explorer**: View weekly prediction scores against actual policy targets
- 🧮 **Feature analysis**: Top features, feature importance comparisons, and SHAP explanations
- 📰 **Media trends**: Topic-level media volume & sentiment time series
- 📊 **Summary tables**: Sample coverage and feature summary tables

---

## Repository Structure

```text
food-policy-media-app/
├─ food_policy_app.py             # Main Shiny application
├─ check_hps_columns.py           # Optional helper to scan outputs for HPS/feature columns
├─ requirements.txt               # Python dependencies
├─ runtime.txt                    # Python version hint (Azure App Service)
├─ .gitignore
├─ .env.example                   # Example environment variables
├─ README.md
├─ outputs_media_policy/          # Model output artifacts (CSV, PNG)
│   ├─ predictions_W.csv
│   ├─ features_W.csv
│   ├─ cv_metrics_summary.csv
│   ├─ rf_top25.csv
│   ├─ lgbm_top25.csv
│   ├─ rf_shap_meanabs.csv
│   ├─ lgbm_shap_meanabs.csv
│   ├─ compare_importance_union.csv
│   ├─ compare_shap_union.csv
│   ├─ gdelt_features_W.csv
│   ├─ table1_sample_coverage_vertical.csv
│   ├─ table2_feature_summary_clean.csv
│   ├─ rf_roc.png
│   ├─ rf_pr.png
│   ├─ lgbm_roc.png
│   ├─ lgbm_pr.png
│   └─ ... (other output images)
```
## Setup & Local Development

1. **Clone repo & create environment**
   ```bash
   git clone https://github.com/<your-username>/food-policy-media-app.git
   cd food-policy-media-app

   # with conda
   conda create -n foodapp python=3.11 -y
   conda activate foodapp

   pip install -r requirements.txt
Point to your outputs

Place your analysis outputs (CSVs/PNGs) inside the outputs_media_policy/ folder
OR set an environment variable to override the path:

bash
Copy
Edit
export FOOD_POLICY_OUTPUT_DIR=/full/path/to/outputs_media_policy
Run locally

bash
Copy
Edit
python -m shiny run --host 0.0.0.0 --port 8000 food_policy_app.py
Visit http://localhost:8000 in your browser.

Deploy to Azure App Service
Create Web App

Publish: Code

Runtime stack: Python 3.11

OS: Linux

App Settings

WEBSITES_PORT = 8000

FOOD_POLICY_OUTPUT_DIR = /home/site/wwwroot/outputs_media_policy

Startup Command

bash
Copy
Edit
python -m shiny run --host 0.0.0.0 --port 8000 food_policy_app.py
Deploy

Push to GitHub and connect via Azure Deployment Center,
or deploy via zip / GitHub Actions.

Confirm

App should be live at https://<your-app-name>.azurewebsites.net

Diagnostics (Overview tab in the app) will show detected files.

Environment Variables
FOOD_POLICY_OUTPUT_DIR: Path where output CSVs/PNGs live.

Default: ./outputs_media_policy in the repo root.

Override with a local .env file or Azure App Service Configuration.

Notes
The Food Insecurity tab is removed in this version (no HPS series present).
Future versions can restore it when HPS data is available.

For reproducibility, either commit outputs_media_policy/ or provide a script to regenerate the artifacts.

Use check_hps_columns.py to confirm which dataset columns are available.

License
Educational / research use only.
© 2025 Jasmine Motupalli
