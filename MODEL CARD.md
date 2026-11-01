# Station/year regression baseline

Educational multi-output random forest. Year and station alone cannot explain changing pollution sources or weather. The UI warns about years outside training range. Units/source permissions still require verification; no drinking-water or safety claims are supported.

Earlier temporal holdout: 473 rows from 2017 onward. R²: O2 0.128, NO3 -0.835, NO2 -0.031, SO4 0.836, PO4 0.526, CL 0.826. Retraining reproduced these figures during this polish pass; no performance improvement is claimed. The training code now also records training-mean baseline MSE for each output. Temporal validation is preferable to a random split here; independent stations and external datasets remain untested.

Run `python train.py /path/to/measurements.csv` (semicolon-delimited) to reproduce on local data, then `streamlit run app.py`. Load only trusted locally trained joblib artifacts.
