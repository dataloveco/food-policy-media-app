"""
Food Policy Media Sentiment Analysis Dashboard
Interactive Shiny App for exploring machine learning predictions
of food insecurity based on media sentiment and coverage
Author: Jasmine Motupalli
"""

from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import base64
import os
import re

# ==========================================
# CONFIGURATION
# ==========================================

EXPECTED_FILES = [
    "predictions_W.csv",
    "features_W.csv",
    "cv_metrics_summary.csv",
    "rf_top25.csv",
    "lgbm_top25.csv",
    "rf_shap_meanabs.csv",
    "lgbm_shap_meanabs.csv",
    "compare_importance_union.csv",
    "compare_shap_union.csv",
    "gdelt_features_W.csv",
    "hps_foodscarce_W.csv",
    "table1_sample_coverage_vertical.csv",
    "table2_feature_summary_clean.csv",
]

def _find_output_dir():
    """
    Heuristic to locate the outputs directory:
    1) FOOD_POLICY_OUTPUT_DIR env var
    2) ./outputs_media_policy
    3) current working dir (if it actually contains the expected files)
    4) any child directory in CWD that contains the expected files
    """
    # 1) env var
    env_dir = os.getenv("FOOD_POLICY_OUTPUT_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # 2) conventional default
    default = Path("outputs_media_policy")
    if default.exists():
        return default

    # 3) CWD contains files
    cwd = Path(".")
    have_any = [ (cwd / f).exists() for f in EXPECTED_FILES ]
    if any(have_any):
        return cwd

    # 4) search children one level deep
    for child in cwd.iterdir():
        if child.is_dir():
            hits = [(child / f).exists() for f in EXPECTED_FILES]
            if any(hits):
                return child

    # fallback to default even if missing (we’ll error verbosely later)
    return default

OUTPUT_DIR = _find_output_dir()

BRAND_COLORS = {
    "purple": "#4A3596",
    "teal": "#27A2AA",
    "coral": "#F2736A",
    "gold": "#FEC84D",
}

# ==========================================
# UTILITIES
# ==========================================

def _parse_dates(df: pd.DataFrame, candidate_cols=("date", "week", "time", "timestamp")):
    """
    Ensure a 'date' column exists and is datetime64[ns].
    If an index looks like dates, use it. Else try common columns. Else try first column.
    """
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        return df

    if df.index.name and df.index.name.lower() in candidate_cols:
        df = df.reset_index()

    for c in candidate_cols:
        if c in df.columns:
            df["date"] = pd.to_datetime(df[c])
            if c != "date":
                # keep original but ensure we have the canonical 'date'
                pass
            return df

    # try first column
    first = df.columns[0]
    try:
        df["date"] = pd.to_datetime(df[first])
        return df
    except Exception:
        return df  # caller will handle error message

def _week_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align to weekly periods starting Monday to ensure GDELT/HPS merge keys match.
    Creates 'week' column (datetime at start of week).
    """
    if "date" not in df.columns:
        return df
    # Robust to tz/na
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W-MON").dt.start_time
    return df

def _find_hps_rate_col(hps: pd.DataFrame) -> str | None:
    """
    Search for a column representing HPS food scarcity rate across possible name variants.
    Returns column name or None.
    """
    patterns = [
        r"food\s*scar(c|s)it(y|e)",      # food scarcity / scarsity typos
        r"food\s*scarce",                # foodscarce
        r"food\s*insufficien",           # food insufficiency
        r"hps.*scarce",                  # hps_foodscarce_...
        r"food.*insecure",               # food_insecure_rate
        r"insufficient.*food",           # narrative variants
        r"food.*hardship",
    ]

    # Prefer likely %/rate columns first (endswith rate/percent)
    rate_like_first = []
    others = []

    for col in hps.columns:
        cu = col.lower()
        if any(re.search(p, cu) for p in patterns):
            if cu.endswith("rate") or "percent" in cu or cu.endswith("%"):
                rate_like_first.append(col)
            else:
                others.append(col)

    candidates = rate_like_first + others

    # If nothing matched, fall back to first numeric column
    if not candidates:
        numeric = hps.select_dtypes(include=[np.number]).columns.tolist()
        return numeric[0] if numeric else None

    return candidates[0]

def _to_percent_if_needed(series: pd.Series) -> pd.Series:
    """
    If values look like proportions in [0,1], convert to percentage.
    Otherwise, leave as-is.
    """
    s = series.dropna()
    if len(s) == 0:
        return series
    if (s.between(0, 1).mean() > 0.9) and (s.max() <= 1.0):
        return series * 100.0
    return series

def encode_image(image_path: Path) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return None

# ==========================================
# DATA LOADING
# ==========================================

def load_data():
    data = {}
    try:
        if not OUTPUT_DIR.exists():
            print(f"[WARN] Output directory not found: {OUTPUT_DIR.resolve()}")
            return data

        def _try_csv(name):
            p = OUTPUT_DIR / name
            return pd.read_csv(p) if p.exists() else None

        # Core sets
        data["predictions"] = _try_csv("predictions_W.csv")
        if data["predictions"] is not None:
            data["predictions"] = _parse_dates(data["predictions"])

        data["features"] = _try_csv("features_W.csv")
        if data["features"] is not None:
            data["features"] = _parse_dates(data["features"])

        data["metrics"] = _try_csv("cv_metrics_summary.csv")

        # Importance / SHAP
        data["rf_importance"] = _try_csv("rf_top25.csv")
        data["lgbm_importance"] = _try_csv("lgbm_top25.csv")
        data["rf_shap"] = _try_csv("rf_shap_meanabs.csv")
        data["lgbm_shap"] = _try_csv("lgbm_shap_meanabs.csv")
        data["importance_compare"] = _try_csv("compare_importance_union.csv")
        data["shap_compare"] = _try_csv("compare_shap_union.csv")

        # GDELT + HPS
        gdelt = _try_csv("gdelt_features_W.csv")
        if gdelt is not None:
            gdelt = _parse_dates(gdelt)
            # If an unnamed index slipped into columns, clean it
            if "Unnamed: 0" in gdelt.columns and "date" not in gdelt.columns:
                gdelt = gdelt.rename(columns={"Unnamed: 0": "date"})
                gdelt = _parse_dates(gdelt)
            data["gdelt"] = _week_align(gdelt)

        hps = _try_csv("hps_foodscarce_W.csv")
        if hps is not None:
            # Normalize dates
            hps = _parse_dates(hps)
            if "Unnamed: 0" in hps.columns and "date" not in hps.columns:
                hps = hps.rename(columns={"Unnamed: 0": "date"})
                hps = _parse_dates(hps)

            # Find the HPS rate column and normalize to percent
            rate_col = _find_hps_rate_col(hps)
            if rate_col is not None:
                hps = hps.copy()
                hps.rename(columns={rate_col: "hps_rate"}, inplace=True)
                if "hps_rate" in hps.columns:
                    hps["hps_rate"] = _to_percent_if_needed(hps["hps_rate"])
            data["hps"] = _week_align(hps)

        # Summary tables
        data["table1"] = _try_csv("table1_sample_coverage_vertical.csv")
        data["table2"] = _try_csv("table2_feature_summary_clean.csv")

        # Attach some diagnostics
        data["_diag"] = {
            "output_dir": str(OUTPUT_DIR.resolve()),
            "files_found": [f for f in EXPECTED_FILES if (OUTPUT_DIR / f).exists()],
        }

    except Exception as e:
        print("[ERROR] load_data:", e)
        import traceback
        traceback.print_exc()

    return data

print("Loading data from:", OUTPUT_DIR.resolve())
DATA = load_data()
print("Datasets loaded:", list(DATA.keys()))

# ==========================================
# UI
# ==========================================

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Overview",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Study Overview"),
                ui.p("This dashboard presents results from a machine learning analysis of media sentiment and food policy."),
                ui.hr(),
                ui.h5("Key Findings:"),
                ui.tags.ul(
                    ui.tags.li("RandomForest ROC AUC: 0.72 ± 0.03"),
                    ui.tags.li("LightGBM ROC AUC: 0.70 ± 0.02"),
                    ui.tags.li("Target prevalence: 2.5%"),
                    ui.tags.li("~240 weekly observations")
                ),
                ui.hr(),
                ui.h5("Data Sources:"),
                ui.tags.ul(
                    ui.tags.li("GDELT media coverage"),
                    ui.tags.li("Census HPS food scarcity"),
                    ui.tags.li("Federal Register policies")
                ),
                ui.hr(),
                ui.h6("Diagnostics"),
                ui.p(f"Output dir: {DATA.get('_diag', {}).get('output_dir','(unknown)')}"),
                ui.p(f"Files detected: {len(DATA.get('_diag', {}).get('files_found', []))} / {len(EXPECTED_FILES)}"),
                width="350px",
                bg="#f8f9fa"
            ),
            ui.row(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("Model Performance Metrics"),
                        ui.output_table("metrics_table"),
                        full_screen=True
                    )
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("Dataset Summary"),
                        ui.output_table("summary_table"),
                        full_screen=True
                    )
                )
            ),
            ui.row(
                ui.column(
                    12,
                    ui.card(
                        ui.card_header("Time Series Overview"),
                        output_widget("timeline_plot"),
                        full_screen=True,
                        height="400px"
                    )
                )
            )
        )
    ),

    ui.nav_panel(
        "Model Predictions",
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("Prediction Scores Over Time"),
                ui.input_select(
                    "model_select",
                    "Select Model:",
                    choices=["RandomForest", "LightGBM", "Both"],
                    selected="Both"
                ),
                output_widget("predictions_plot"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Model Performance Curves"),
                ui.input_radio_buttons(
                    "curve_type",
                    "Curve Type:",
                    choices=["ROC", "Precision-Recall"],
                    selected="ROC"
                ),
                ui.output_ui("performance_curves"),
                full_screen=True
            ),
            width="50%"
        )
    ),

    ui.nav_panel(
        "Feature Analysis",
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("Feature Importance Comparison"),
                ui.input_slider(
                    "n_features",
                    "Number of features to display:",
                    min=5,
                    max=25,
                    value=15,
                    step=5
                ),
                output_widget("importance_comparison"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("SHAP Value Comparison"),
                output_widget("shap_comparison"),
                full_screen=True
            ),
            width="50%"
        ),
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Feature Statistics"),
                    ui.output_data_frame("feature_stats"),
                    full_screen=True
                )
            )
        )
    ),

    ui.nav_panel(
        "Media Trends",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Media Coverage Controls"),
                ui.input_select(
                    "media_topic",
                    "Select Topic:",
                    choices=[
                        "Food insecurity",
                        "SNAP",
                        "WIC",
                        "Food pantry",
                        "School meals",
                        "Food prices"
                    ],
                    selected="Food insecurity"
                ),
                ui.input_radio_buttons(
                    "media_metric",
                    "Metric:",
                    choices=["Volume", "Sentiment", "Both"],
                    selected="Both"
                ),
                ui.hr(),
                ui.input_date_range(
                    "date_range",
                    "Date Range:",
                    start="2021-01-01",
                    end="2025-08-31"
                ),
                width="300px"
            ),
            ui.card(
                ui.card_header("Media Coverage Analysis"),
                output_widget("media_trends"),
                full_screen=True,
                height="500px"
            )
        )
    ),

    ui.nav_panel(
        "Documentation",
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("About This Dashboard"),
                ui.markdown("""
                ### Research Question
                Can media sentiment and topical volume from publicly available sources be used to predict 
                near-term changes in household food insecurity in the United States?

                ### Methodology
                - **Data Sources**: GDELT media database, Census HPS, Federal Register
                - **Models**: Random Forest and LightGBM classifiers
                - **Validation**: 5-fold time series cross-validation
                - **Target**: Policy events within 14-day horizon

                ### Key Findings
                1. Media volume more predictive than sentiment alone
                2. Food prices and SNAP coverage are strongest predictors
                3. Both models achieve ~70% ROC AUC
                4. Short-term lags (1-2 weeks) most informative

                ### Author
                Jasmine Motupalli  
                Daniels College of Business, University of Denver  
                FIN-6305: Applied Quantitative Methods
                """),
                full_screen=True
            ),
            width="100%"
        )
    ),

    title="Food Policy Media Sentiment Analysis Dashboard",
    id="navbar",
    inverse=True,
    bg=BRAND_COLORS["purple"]
)

# ==========================================
# SERVER
# ==========================================

def server(input: Inputs, output: Outputs, session: Session):
    
    @output
    @render.text
    def out_dir():
        return f"Output dir: {DATA.get('_diag', {}).get('output_dir','(unknown)')}"
        
    @output
    @render.text
    def files_found():
        return f"Files detected: {len(DATA.get('_diag', {}).get('files_found', []))} / {len(EXPECTED_FILES)}"

    @output
    @render.table
    def metrics_table():
        if DATA.get("metrics") is not None:
            return DATA["metrics"].round(3)
        return pd.DataFrame({"Note": ["No metrics data available"]})

    @output
    @render.table
    def summary_table():
        if DATA.get("table1") is not None:
            return DATA["table1"]
        return pd.DataFrame({"Note": ["No summary data available"]})

    @output
    @render_widget
    def timeline_plot():
        df = DATA.get("predictions")
        if df is None or "date" not in df.columns:
            return go.Figure().add_annotation(text="No predictions data available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Model Predictions", "Actual Target"), vertical_spacing=0.15)
        if "rf_score" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["rf_score"], name="RF Score", line=dict(color=BRAND_COLORS["purple"])), row=1, col=1)
        if "lgbm_score" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["lgbm_score"], name="LGBM Score", line=dict(color=BRAND_COLORS["teal"])), row=1, col=1)
        if "y_true" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["y_true"], name="Target", line=dict(color=BRAND_COLORS["coral"])), row=2, col=1)
        fig.update_layout(height=400, showlegend=True, title="Model Predictions and Target Over Time")
        return fig

    @output
    @render_widget
    def predictions_plot():
        df = DATA.get("predictions")
        if df is None or "date" not in df.columns:
            return go.Figure().add_annotation(text="No predictions data available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig = go.Figure()
        if input.model_select() in ["RandomForest", "Both"] and "rf_score" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["rf_score"], name="RandomForest", line=dict(color=BRAND_COLORS["purple"], width=2)))
        if input.model_select() in ["LightGBM", "Both"] and "lgbm_score" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["lgbm_score"], name="LightGBM", line=dict(color=BRAND_COLORS["teal"], width=2)))
        if "y_true" in df.columns:
            fig.add_trace(go.Bar(x=df["date"], y=df["y_true"], name="Target Events", marker_color=BRAND_COLORS["gold"], opacity=0.3))
        fig.update_layout(title="Prediction Scores Over Time", xaxis_title="Date", yaxis_title="Score", hovermode="x unified")
        return fig

    @output
    @render.ui
    def performance_curves():
        curve = "roc" if input.curve_type() == "ROC" else "pr"
        images = []
        for model, label in [("rf", "Random Forest"), ("lgbm", "LightGBM")]:
            p = OUTPUT_DIR / f"{model}_{curve}.png"
            enc = encode_image(p) if p.exists() else None
            if enc:
                images.append(ui.column(6, ui.h5(label), ui.tags.img(src=enc, style="width: 100%; height: auto;")))
        return ui.row(*images) if images else ui.p("Performance curves not available")

    @output
    @render_widget
    def importance_comparison():
        df = DATA.get("importance_compare")
        if df is None:
            return go.Figure().add_annotation(text="No importance comparison data available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        df = df.head(input.n_features())
        fig = go.Figure()
        if "RandomForest_norm" in df.columns:
            fig.add_trace(go.Bar(y=df["feature_pretty"], x=df["RandomForest_norm"], name="RandomForest", orientation="h", marker_color=BRAND_COLORS["purple"]))
        if "LightGBM_norm" in df.columns:
            fig.add_trace(go.Bar(y=df["feature_pretty"], x=df["LightGBM_norm"], name="LightGBM", orientation="h", marker_color=BRAND_COLORS["teal"]))
        fig.update_layout(title=f"Top {input.n_features()} Features - Importance Comparison",
                          xaxis_title="Normalized Importance", yaxis_title="", barmode="group",
                          height=max(400, input.n_features() * 30))
        return fig

    @output
    @render_widget
    def shap_comparison():
        df = DATA.get("shap_compare")
        if df is None:
            return go.Figure().add_annotation(text="No SHAP comparison data available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        df = df.head(15)
        fig = go.Figure()
        if "RandomForest_norm" in df.columns:
            fig.add_trace(go.Bar(y=df["feature_pretty"], x=df["RandomForest_norm"], name="RandomForest", orientation="h", marker_color=BRAND_COLORS["purple"]))
        if "LightGBM_norm" in df.columns:
            fig.add_trace(go.Bar(y=df["feature_pretty"], x=df["LightGBM_norm"], name="LightGBM", orientation="h", marker_color=BRAND_COLORS["teal"]))
        fig.update_layout(title="Mean |SHAP| Value Comparison", xaxis_title="Normalized SHAP Value", yaxis_title="", barmode="group", height=500)
        return fig

    @output
    @render.data_frame
    def feature_stats():
        df = DATA.get("table2")
        if df is None:
            return render.DataGrid(pd.DataFrame({"Note": ["No feature statistics available"]}))
        return render.DataGrid(df.round(3), filters=True)  # modern DataGrid pattern

    @output
    @render_widget
    def media_trends():
        gdelt = DATA.get("gdelt")
        if gdelt is None or "date" not in gdelt.columns:
            return go.Figure().add_annotation(text="No media trends data available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        topic_map = {
            "Food insecurity": '("food insecurity" OR "food insufficiency")',
            "SNAP": '(SNAP OR "Supplemental Nutrition Assistance Program")',
            "WIC": '(WIC OR "Women, Infants, and Children")',
            "Food pantry": '("food pantry" OR "food bank")',
            "School meals": '("school lunch" OR "school breakfast")',
            "Food prices": '("food prices" OR "grocery inflation")',
        }
        query = topic_map.get(input.media_topic(), "")
        vol_col = f"vol[{query}]"
        sent_col = f"sent[{query}]"

        start_date = pd.to_datetime(input.date_range()[0])
        end_date = pd.to_datetime(input.date_range()[1])
        mask = (gdelt["date"] >= start_date) & (gdelt["date"] <= end_date)
        df = gdelt.loc[mask].copy()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        missing = []
        if input.media_metric() in ["Volume", "Both"]:
            if vol_col in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df[vol_col], name="Volume",
                                         line=dict(color=BRAND_COLORS["purple"], width=2)),
                              secondary_y=False)
            else:
                missing.append("Volume")
        if input.media_metric() in ["Sentiment", "Both"]:
            if sent_col in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df[sent_col], name="Sentiment",
                                         line=dict(color=BRAND_COLORS["teal"], width=2)),
                              secondary_y=True)
            else:
                missing.append("Sentiment")

        if missing:
            fig.add_annotation(text=f"Missing columns for: {', '.join(missing)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Volume", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment", secondary_y=True)
        fig.update_layout(title=f"Media Coverage: {input.media_topic()}", hovermode="x unified")
        return fig

    @output
    @render_widget
    def hps_trends():
        hps = DATA.get("hps")
        if hps is None or "date" not in hps.columns:
            return go.Figure().add_annotation(text="No HPS data available - ensure hps_foodscarce_W.csv is present and readable",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        rate_col = "hps_rate" if "hps_rate" in hps.columns else _find_hps_rate_col(hps)
        if rate_col is None:
            return go.Figure().add_annotation(text=f"HPS loaded but no rate column detected. Columns: {', '.join(hps.columns)}",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        y = hps[rate_col].astype(float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hps["date"], y=y, mode="lines+markers",
                                 name="Food Scarcity Rate", line=dict(color=BRAND_COLORS["coral"], width=2),
                                 marker=dict(size=4)))
        if y.notna().sum() > 1:
            # simple linear trend over observed points
            idx = np.arange(len(y))
            mask = y.notna()
            z = np.polyfit(idx[mask], y[mask], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=hps["date"], y=p(idx), mode="lines",
                                     name="Trend", line=dict(color=BRAND_COLORS["gold"], width=2, dash="dash")))
        fig.update_layout(title="Household Food Scarcity Rate Over Time",
                          xaxis_title="Date", yaxis_title="Food Scarcity Rate (%)", hovermode="x unified")
        return fig

    @output
    @render.ui
    def hps_stats():
        hps = DATA.get("hps")
        if hps is None:
            return ui.HTML("<div style='padding:10px;'>No HPS data loaded</div>")
        rate_col = "hps_rate" if "hps_rate" in hps.columns else _find_hps_rate_col(hps)
        if rate_col is None:
            return ui.HTML(f"<div style='padding:10px;'>No rate column found. Available: {', '.join(hps.columns)}</div>")

        rate = pd.to_numeric(hps[rate_col], errors="coerce")
        current_rate = float(rate.dropna().iloc[-1]) if rate.dropna().size else np.nan
        mean_rate = float(rate.mean()) if rate.notna().any() else np.nan
        std_rate  = float(rate.std())  if rate.notna().any() else np.nan
        min_rate  = float(rate.min())  if rate.notna().any() else np.nan
        max_rate  = float(rate.max())  if rate.notna().any() else np.nan
        change    = float(current_rate - rate.dropna().iloc[0]) if rate.dropna().size > 1 else np.nan

        def fmt(x): return "—" if pd.isna(x) else f"{x:.2f}%"
        color = "red" if (not pd.isna(change) and change > 0) else "green"

        html = f"""
        <div style="padding:10px;">
            <h6>Current Rate:</h6>
            <h3 style="color:{BRAND_COLORS['coral']};">{fmt(current_rate)}</h3>
            <hr>
            <p><strong>Mean:</strong> {fmt(mean_rate)}</p>
            <p><strong>Std Dev:</strong> {fmt(std_rate)}</p>
            <p><strong>Min:</strong> {fmt(min_rate)}</p>
            <p><strong>Max:</strong> {fmt(max_rate)}</p>
            <hr>
            <p><strong>Change from start:</strong> <span style="color:{color};">{fmt(change)}</span></p>
            <p style="font-size:0.8em;color:gray;">Column used: {rate_col}</p>
        </div>
        """
        return ui.HTML(html)

    @output
    @render_widget
    def correlation_heatmap():
        gdelt = DATA.get("gdelt")
        hps = DATA.get("hps")
        if gdelt is None or hps is None:
            return go.Figure().add_annotation(text="GDELT or HPS data not available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        rate_col = "hps_rate" if "hps_rate" in hps.columns else _find_hps_rate_col(hps)
        if rate_col is None:
            return go.Figure().add_annotation(text="No HPS rate column detected",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        # work on week-aligned keys
        if "week" not in gdelt.columns:
            gdelt = _week_align(gdelt)
        if "week" not in hps.columns:
            hps = _week_align(hps)

        vol_cols = [c for c in gdelt.columns if c.startswith("vol[")]
        if not vol_cols:
            return go.Figure().add_annotation(text="No media volume columns found in GDELT",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        merged = pd.merge(
            gdelt[["week"] + vol_cols],
            hps[["week", rate_col]].rename(columns={rate_col: "hps_rate"}),
            on="week",
            how="inner",
        )

        if merged.empty:
            return go.Figure().add_annotation(text="No overlapping weeks between GDELT and HPS (after week alignment)",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        corr = merged[vol_cols].corrwith(merged["hps_rate"])
        clean_names = (
            corr.index
                .str.replace(r"^vol\[", "", regex=True)
                .str.replace(r"\]$", "", regex=True)
                .str.replace('"', "", regex=False)
                .str.replace("(", "", regex=False)
                .str.replace(")", "", regex=False)
                .str.replace(" OR ", " / ", regex=False)
                .str.slice(0, 60)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=corr.values,
                y=clean_names,
                orientation="h",
                marker_color=[BRAND_COLORS["purple"] if x > 0 else BRAND_COLORS["coral"] for x in corr.values],
                text=[f"{x:.3f}" for x in corr.values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Correlation with HPS Food Scarcity Rate (weekly aligned)",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Media Coverage Topic",
            height=max(400, 20 * len(clean_names)),
            xaxis=dict(range=[-1, 1]),
        )
        return fig

# ==========================================
# CREATE AND RUN APP
# ==========================================
app = App(app_ui, server)
