# StreamFlowAI — ANN Streamflow Forecasting

This project follows the agreed reference workflow:
- Daily data: 2003–2012
- Training: 2003–2007
- Validation: 2008–2012
- No test set
- Inputs: lagged rainfall + lagged streamflow
- Main model: standard feed-forward ANN
- Hyperparameter tuning: required
- Primary metric: RMSE
- Peak threshold: 1500 ft³/s
- Peak, non-peak and overall RMSE are reported
- Linear Regression and Random Forest are comparison baselines

DATA NOTE
The included downloader obtains the public MACH dataset and extracts the Leaf River
USGS station 02472000. MACH is not claimed to be the exact raw CSV used by Xiao & You;
the Stanford report does not publish that raw CSV. MACH covers 1980–2023 and provides
daily precipitation plus USGS streamflow. See SOURCE_NOTE.md.

FIRST RUN
1. Create/activate a Python environment.
2. pip install -r requirements.txt
3. python download_dataset.py
4. python main.py

The downloader creates data/leaf_river_2003_2012.csv.
