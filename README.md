# Water Pollutant Prediction

A multi-output regression project and Streamlit interface for estimating O2, NO3, NO2, SO4, PO4 and CL values from station ID and year.

## What it demonstrates

- Input schema validation and missing-target handling.
- A shared station-encoding pipeline for training and inference.
- A random-forest multi-output regression baseline.
- A chronological holdout across years and per-pollutant MSE/R² reports.
- An interface restricted to stations observed during training.

## Run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py /path/to/PB_All_2000_2021.csv
.venv/bin/streamlit run app.py
```

Supply a local semicolon-delimited CSV with `id`, `date`, and the six pollutant columns above. Dates are parsed day-first. The training split holds out the latest years; at least three distinct years are required. The model artifact and evaluation JSON are saved to ignored `data/`.

## Portfolio preparation

The original exploration had an undefined training target and inconsistent one-hot encoding. The maintained `train.py`/`app.py` path fixes those issues and uses temporal validation. `exploration.ipynb` retains the output-cleared exploratory work with broken assistant-placeholder cells removed. Local datasets and pretrained pickle files are excluded.

## Limitations

Year and station alone omit weather, flow, industrial activity and many other causes of water quality changes. Random forests cannot reliably extrapolate future trends. Out-of-range year inputs display a warning. Units must come from the source dataset; the project does not establish drinking-water safety or regulatory compliance. No operational forecasting accuracy is claimed.
