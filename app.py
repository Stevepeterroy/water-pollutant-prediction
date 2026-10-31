from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

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
    st.dataframe(pd.DataFrame({"Pollutant": artifact["pollutants"], "Estimate (source dataset units)": prediction}))
