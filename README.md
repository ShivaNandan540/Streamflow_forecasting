# StreamFlowAI — ANN Streamflow Forecasting

This project follows the agreed reference workflow:
- Daily data: 2003–2012
- Training: 2003–2007
- Validation: 2008–2012
- No test set
- Inputs: lagged rainfall + lagged streamflow
- Models: standard feed-forward ANN and paper-style LSTM
- Feed-forward setup: one sigmoid hidden layer, linear output, Adam at 0.009
- Lag search: rainfall and streamflow lags from 2 through 7 days
- Hidden sizes searched: 8, 12, 16, 24, and 31 (under the paper's limit of 32)
- Training: 200 epochs per tuning configuration and for the final ANN
- Scaling: divide rainfall and streamflow by their respective training-period maxima
- LSTM setup: equal 5-day rainfall/streamflow sequence, 12 hidden units, learning rate 0.001
- Primary metric: RMSE
- Peak threshold: 1500 ft³/s
- Peak, non-peak and overall RMSE are reported
- Linear Regression and Random Forest are comparison baselines
- Negative flow predictions are clipped to zero before evaluation and output

DATA NOTE
The included downloader obtains the public MACH dataset and extracts the Leaf River
USGS station 02472000. MACH is not claimed to be the exact raw CSV used by Xiao & You;
the Stanford report does not publish that raw CSV. MACH covers 1980–2023 and provides
daily precipitation plus USGS streamflow. See SOURCE_NOTE.md.
Because the original CSV is unavailable, rainfall and streamflow scaling maxima are
estimated from the 2003-2007 training split. This preserves the paper's divide-by-maximum
method without assuming its reported maxima apply exactly to the substitute dataset.
The paper does not specify its optimizer or batch size; this implementation uses Adam
and batch size 32, so those details cannot be claimed as an exact match.
The LSTM uses a custom cell with the paper's multi-prior-cell-state gate equations.
Its validation predictions are written separately because its fixed 5-day sequence
does not necessarily align row-for-row with the ANN's tuned lag configuration.

FIRST RUN
1. Create/activate a Python environment.
2. pip install -r requirements.txt
3. python download_dataset.py
4. python main.py

The downloader creates data/leaf_river_2003_2012.csv.
