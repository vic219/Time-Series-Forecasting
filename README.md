
# Time Series Forecasting – Daily Delhi Temperature (ARIMA, Prophet, LSTM)

This repository contains a complete time series forecasting project on **daily mean temperature in Delhi**.  
It compares three modeling approaches:

- **ARIMA** – classical statistical baseline  
- **Facebook Prophet** – additive model with explicit trend and seasonality  
- **LSTM** – deep learning sequence model  

> **Note:** The dataset (CSV) is **not included** in this repository due to licensing.  
> Instructions to download it yourself are provided in the **Dataset** section.

---

## 1. Project Overview

The goal of this project is to:

- Load and preprocess a real-world time series dataset (daily climate data for Delhi)
- Perform exploratory data analysis (EDA)
- Train and compare three forecasting models:
  - ARIMA  
  - Prophet  
  - LSTM  
- Evaluate models using:
  - **MAE** – Mean Absolute Error  
  - **RMSE** – Root Mean Squared Error  
  - **MAPE** – Mean Absolute Percentage Error
- Visualize and interpret the forecasts and residuals

This project is designed to be **Google Colab–friendly** and also runnable locally via Jupyter Notebook.

---

## 2. Repository Structure

```text
.
├── notebooks/
│   └── Time_Series_Forecasting.ipynb
├── data/
│   ├── raw/          # Place the original CSV from Kaggle here (NOT tracked in repo)
│   └── processed/    # Optional processed train/test CSVs
├── plots/            # Saved EDA and forecast plots (PNG)
├── output/           # Forecast CSVs and metrics (MAE, RMSE, MAPE)
├── report/
│   └── Time_Series_Forecasting_Report.pdf
├── requirements.txt
└── README.md
````

---

## 3. Dataset (Not Included in Repo)

The project uses a **publicly available daily climate time series dataset for Delhi (2013–2017)**.

Typical columns include:

* `date` – date of observation
* `meantemp` – daily mean temperature (°C)
* `humidity`
* `wind_speed`
* `meanpressure`

### 3.1 How to Download the Dataset (Manual Kaggle Download)

1. Go to Kaggle and search for:
   **“Daily Delhi Climate Time Series Data”**
2. Download the dataset.
3. Create the folder:

```text
data/raw/
```

4. Place the CSV file into:

```text
data/raw/DailyDelhiClimate.csv
```

5. If the file name is different, update the notebook accordingly.

---

### 3.2 Optional: Download via `kagglehub` Inside the Notebook

```python
import kagglehub
import os
import shutil

# Download dataset from Kaggle
path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")
print("Downloaded to:", path)

# Copy one CSV file into data/raw/
os.makedirs("data/raw", exist_ok=True)
shutil.copyfile(
    os.path.join(path, "DailyDelhiClimateTrain.csv"),
    "data/raw/DailyDelhiClimate.csv"
)
```

---

## 4. Installation & Environment

It is recommended to use a virtual environment.

### 4.1 Create and Activate Virtual Environment

```bash
# Create venv
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 4.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages include:

* pandas, numpy
* scikit-learn
* statsmodels
* prophet + cmdstanpy
* tensorflow
* matplotlib, seaborn
* kagglehub (optional for dataset download)
* jupyter, notebook, ipykernel

---

## 5. How to Run the Notebook

### 5.1 Run in Google Colab

1. Upload/clone the repository into your Drive.
2. Open:

```
notebooks/Time_Series_Forecasting.ipynb
```

3. Ensure the dataset exists at:

```
data/raw/DailyDelhiClimate.csv
```

You may upload it manually OR use the KaggleHub code cell.

4. Run all cells in order:

* Data loading & preprocessing
* EDA
* Model training (ARIMA, Prophet, LSTM)
* Forecasting & evaluation
* Plot saving & output CSV generation

Generated results appear in:

* `plots/`
* `output/`

---

### 5.2 Run Locally with Jupyter

```bash
jupyter notebook
```

Then open the notebook and ensure the dataset CSV is present before running.

---

## 6. Modeling Details

### 6.1 Preprocessing & EDA

* Convert date to datetime, set index
* Resample to daily frequency
* Handle missing values
* Generate:

  * Time series plots
  * Rolling mean & std
  * Seasonal decomposition

### 6.2 ARIMA

* Make data stationary (differencing)
* Use ACF/PACF to choose p, q
* Fit ARIMA model
* Forecast test set

### 6.3 Prophet

* Convert data to `ds` and `y` format
* Fit model with yearly seasonality
* Forecast and align with test period

### 6.4 LSTM

* Normalize data
* Create sliding windows
* Build LSTM model
* Invert normalization after prediction

---

## 7. Evaluation & Metrics

Models evaluated using:

* **MAE**
* **RMSE**
* **MAPE**

Metrics are saved in:

```
output/metrics_summary.csv
```

---

## 8. Plots & Outputs

Saved by the notebook:

* EDA plots → `plots/eda/`
* Forecast vs actual plots → `plots/forecasts/`
* Residual plots → `plots/residuals/`
* Forecast CSVs → `output/`
* Metrics summary → `output/metrics_summary.csv`

---

## 9. Report

A full project report is available in:

```
report/Time_Series_Forecasting_Report.pdf
```

---

## 10. License & Acknowledgements

* The original dataset belongs to the Kaggle author(s).
* This project is intended for education, learning, and portfolio use.

```

---

### ✅ FINAL ANSWER  
**Yes — after replacing your version with THIS corrected version, your README will be perfectly formatted, clean, and fully ready for GitHub.**
```
