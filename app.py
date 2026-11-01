from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import json

st.set_page_config(page_title="Water Pollutant Prediction", page_icon="💧")
st.markdown("""<style>
.block-container{max-width:1080px;padding-top:3rem}h1,h2,h3{font-family:Georgia,serif!important;letter-spacing:-.035em}h1{font-size:3rem!important}div[data-testid="stForm"]{background:#fffdf7;border:1px solid #d6d8cc;border-radius:18px;padding:26px}div[data-testid="stAlert"]{border-radius:12px}button{border-radius:9px!important}div[data-testid="stFileUploader"]{border:1px dashed #aeb9aa;border-radius:14px;padding:10px}div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}@media(max-width:600px){h1{font-size:2.1rem!important}.block-container{padding-top:1.5rem}}
</style>""", unsafe_allow_html=True)
st.title("Water Pollutant Regression Baseline")
st.caption("Educational station/year model. Estimates are not a water safety assessment or a validated forecast.")
path = Path(__file__).parent / "data/water-model.joblib"
if not path.exists():
    st.info("Train your local model first: python train.py /path/to/measurements.csv")
    st.stop()
artifact = joblib.load(path)
low, high = artifact["training_years"]
st.write(f"Training observations: {low}–{high}")
station = st.selectbox("Station ID", artifact["stations"])
year = st.number_input("Year", min_value=1900, max_value=2100, value=high)
if year < low or year > high:
    st.warning("This year is outside the training range. Random forests do not model long-term trends reliably; treat this as an extrapolation limitation.")

if st.button("Estimate pollutants"):
    prediction = artifact["model"].predict(pd.DataFrame([{"year": year, "id": station}]))[0]
    df_result = pd.DataFrame({"Pollutant": artifact["pollutants"], "Estimate (source dataset units)": prediction})
    st.dataframe(df_result)

metrics_path = path.with_suffix(".metrics.json")
if metrics_path.exists():
    st.subheader("Temporal validation by pollutant")
    metrics = json.loads(metrics_path.read_text())
    st.caption(f"Held out {metrics['test_rows']} rows from {metrics['temporal_test_start']} onward. Negative R² means worse than the test-set mean reference.")
    st.dataframe(pd.DataFrame(metrics["pollutants"]).T)
st.info("Units have not been verified against the original dataset documentation. Do not compare these values with drinking-water limits.")
