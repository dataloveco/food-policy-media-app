# Food Policy Media Sentiment Analysis Dashboard

An interactive [Shiny for Python](https://shiny.posit.co/py/) application for exploring the relationship between **media coverage** (volume and sentiment) and **food policy actions** in the U.S.

Built as part of **FIN-6305 Quantitative Methods III**, University of Denver.

## ✨ Features

- **📈 Model Performance Overview**: Random Forest and LightGBM cross-validated ROC/PR metrics
- **🔎 Prediction Explorer**: View weekly prediction scores against actual policy targets
- **🧮 Feature Analysis**: Top features, feature importance comparisons, and SHAP explanations
- **📰 Media Trends**: Topic-level media volume & sentiment time series
- **📊 Summary Tables**: Sample coverage and feature summary tables

## 📁 Repository Structure

```
food-policy-media-app/
├── LICENSE
├── README.md
├── food_policy_app.py                    # Main Shiny application
├── requirements.txt                      # Python dependencies
├── runtime.txt                          # Python version hint (Azure App Service)
└── outputs_media_policy/               # Model output artifacts
    ├── check_hps_columns.py            # Helper script to scan outputs for columns
    ├── cv_metrics_summary.csv          # Cross-validation metrics summary
    ├── predictions_W.csv               # Model predictions data
    ├── features_W.csv                  # Feature data
    ├── gdelt_features_W.csv           # GDELT feature data
    ├── model_dataset_W.csv            # Complete model dataset
    ├── hps_foodscarce_W.csv          # Food scarcity data
    ├── feature_name_map.csv          # Feature name mappings
    │
    ├── Random Forest outputs:
    │   ├── rf_top25.csv              # Top 25 RF features
    │   ├── rf_shap_meanabs.csv       # RF SHAP values
    │   ├── rf_roc.png               # RF ROC curve
    │   ├── rf_pr.png                # RF Precision-Recall curve
    │   ├── rf_top25.png             # RF feature importance plot
    │   ├── rf_shap_bar.png          # RF SHAP bar plot
    │   ├── rf_shap_bar_norm.png     # RF SHAP normalized bar plot
    │   ├── rf_shap_swarm.png        # RF SHAP swarm plot
    │   └── rf_shap_swarm_norm.png   # RF SHAP normalized swarm plot
    │
    ├── LightGBM outputs:
    │   ├── lgbm_top25.csv           # Top 25 LightGBM features
    │   ├── lgbm_shap_meanabs.csv    # LightGBM SHAP values
    │   ├── lgbm_roc.png             # LightGBM ROC curve
    │   ├── lgbm_pr.png              # LightGBM Precision-Recall curve
    │   ├── lgbm_top25.png           # LightGBM feature importance plot
    │   ├── lgbm_shap_bar.png        # LightGBM SHAP bar plot
    │   └── lgbm_shap_swarm.png      # LightGBM SHAP swarm plot
    │
    ├── Model comparison outputs:
    │   ├── compare_importance_union.csv     # Feature importance comparison
    │   ├── compare_shap_union.csv          # SHAP comparison
    │   ├── compare_importance_rf_vs_lgbm.png # Importance comparison plot
    │   └── compare_shap_rf_vs_lgbm.png     # SHAP comparison plot
    │
    └── Summary tables:
        ├── table1_sample_coverage.csv
        ├── table1_sample_coverage_vertical.csv
        ├── table2_feature_summary.csv
        └── table2_feature_summary_clean.csv
```

## 🚀 Setup & Local Development

### 1. Clone Repository & Create Environment

```bash
git clone https://github.com/<your-username>/food-policy-media-app.git
cd food-policy-media-app

# Create conda environment
conda create -n foodapp python=3.11 -y
conda activate foodapp

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Output Path

Place your analysis outputs (CSVs/PNGs) inside the `outputs_media_policy/` folder, or set an environment variable to override the path:

```bash
export FOOD_POLICY_OUTPUT_DIR=/full/path/to/outputs_media_policy
```

### 3. Run Locally

```bash
python -m shiny run --host 0.0.0.0 --port 8000 food_policy_app.py
```

Visit [http://localhost:8000](http://localhost:8000) in your browser.

## ☁️ Deploy to Azure App Service

### Create Web App
- **Publish**: Code
- **Runtime stack**: Python 3.11
- **OS**: Linux

### App Settings
```
WEBSITES_PORT = 8000
FOOD_POLICY_OUTPUT_DIR = /home/site/wwwroot/outputs_media_policy
```

### Startup Command
```bash
python -m shiny run --host 0.0.0.0 --port 8000 food_policy_app.py
```

### Deploy
- Push to GitHub and connect via Azure Deployment Center, or
- Deploy via zip upload / GitHub Actions

### Confirm Deployment
Your app should be live at `https://<your-app-name>.azurewebsites.net`

## ⚙️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FOOD_POLICY_OUTPUT_DIR` | Path where output CSVs/PNGs are located | `./outputs_media_policy` |

Override with a local `.env` file or Azure App Service Configuration.

## 📝 Notes

- The Food Insecurity tab has been removed in this version (no HPS series present)
- Future versions can restore it when HPS data becomes available
- For reproducibility, either commit `outputs_media_policy/` or provide a script to regenerate the artifacts
- Use `check_hps_columns.py` to confirm which dataset columns are available

## 📄 License

Educational / research use only.  
© 2025 Jasmine Motupalli
