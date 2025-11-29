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
│   └── processed/    # Optional: processed train/test CSVs
├── plots/            # Saved EDA and forecast plots (PNG)
├── output/           # Forecast CSVs and metrics (MAE, RMSE, MAPE)
├── report/
│   └── Time_Series_Forecasting_Report.pdf
├── requirements.txt
└── README.md

**## 3. Dataset (Not Included in Repo)**

The project uses a publicly available daily climate time series dataset for Delhi (around 2013–2017).
It typically includes columns such as:

date – date of observation

meantemp – daily mean temperature (°C)

humidity, wind_speed, meanpressure (optional features)

3.1 How to Download the Dataset (Kaggle – Manual)

Go to Kaggle and search for:

Daily Delhi Climate Time Series Data

Download the CSV file from the dataset page.

Create the following folders in your local repo:

data/
  raw/


Place the downloaded CSV into:

data/raw/DailyDelhiClimate.csv


If the file name is different, update the path in the notebook accordingly.

**## 3.2 Option: Download via kagglehub in the Notebook
**
If you have a Kaggle API token configured, you can download the dataset programmatically.
Example pattern inside the notebook:

import kagglehub
import os
import shutil

# Download dataset from Kaggle
path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")
print("Downloaded to:", path)

# Copy one of the CSVs into data/raw/
os.makedirs("data/raw", exist_ok=True)
shutil.copyfile(
    os.path.join(path, "DailyDelhiClimateTrain.csv"),
    "data/raw/DailyDelhiClimate.csv"
)


Adjust filenames if needed based on the dataset contents.

**## 4. Installation & Environment
**
It is recommended to use a virtual environment.

4.1 Create and Activate Virtual Environment (Local)
# Create venv (example for Python 3)
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

## 4.2 Install Dependencies

Install all required libraries with:

pip install -r requirements.txt


Key packages include:

pandas, numpy

scikit-learn

statsmodels

prophet and cmdstanpy

tensorflow

matplotlib, seaborn

kagglehub (for dataset download, optional)

jupyter, notebook, ipykernel

**## 5. How to Run the Notebook
** 5.1 Run in Google Colab

Upload this repository (or clone it into your Google Drive).

Open notebooks/Time_Series_Forecasting.ipynb in Google Colab.

Make sure your dataset CSV is available at data/raw/DailyDelhiClimate.csv:

Either upload it manually to that path, or

Use the kagglehub download code cell to fetch and copy it there.

Run all cells from top to bottom:

Data loading & preprocessing

EDA & visualization

Model training (ARIMA, Prophet, LSTM)

Forecasting & evaluation

Plot saving & output CSV generation

Generated files will be saved to:

plots/ – visualizations (EDA, forecasts, residuals, comparisons)

output/ – forecast values and metrics (.csv)

5.2 Run Locally with Jupyter

Create and activate a virtual environment and install the requirements.

Start Jupyter:

jupyter notebook


Open notebooks/Time_Series_Forecasting.ipynb.

Ensure data/raw/DailyDelhiClimate.csv exists, then run the notebook.

## 6. Modeling Details
6.1 Preprocessing & EDA

Typical preprocessing steps in the notebook:

Convert date column to datetime and set it as index.

Sort by date and resample to daily frequency (if necessary).

Handle missing values (e.g. interpolation).

Generate:

Time series plot of meantemp

Rolling mean and standard deviation

Seasonal decomposition (trend, seasonal, residual)

6.2 ARIMA Model

Make the series approximately stationary using differencing if needed.

Select ARIMA orders (p, d, q) via ACF/PACF and/or information criteria.

Fit the model on the training data.

Forecast over the test period.

6.3 Prophet Model

Prepare data in Prophet’s format with columns ds (date) and y (temperature).

Fit model with yearly seasonality (and optionally weekly/daily components).

Generate a future dataframe for the forecast horizon.

Align forecasts with the test period and evaluate.

6.4 LSTM Model

Scale the target variable using MinMaxScaler.

Create supervised sequences using a sliding window of past observations.

Define and train an LSTM-based neural network (with optional dropout & early stopping).

Invert scaling to get predictions back in original temperature units.

## 7. Evaluation & Metrics

All models are evaluated on a held-out test period using:

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

MAPE – Mean Absolute Percentage Error

Metrics for each model are stored in an output CSV (e.g. output/metrics_summary.csv), and also printed in the notebook with comparison plots.

## 8. Plots & Outputs

The notebook saves key results as:

EDA plots (time series, decomposition, rolling stats) → plots/eda/

Forecast vs actual plots for ARIMA / Prophet / LSTM → plots/forecasts/

Residual plots for each model → plots/residuals/

Forecast CSVs for each model → output/

Aggregated metrics CSV → output/metrics_summary.csv

You can commit the plots/ and output/ folders (excluding very large files) to show results on GitHub.

## 9. Report

A detailed written summary of the modeling process and results is provided in:

report/Time_Series_Forecasting_Report.pdf


The report includes:

Problem statement and motivation

Data description

Methodology for each model

Quantitative results (MAE, RMSE, MAPE)

Visual analysis and discussion

Conclusion and potential improvements

## 10. License & Acknowledgements

The original climate dataset is provided by the dataset author(s) on Kaggle.
Please check the dataset license on Kaggle before using it for anything beyond educational or personal use.

This repository is intended for learning, experimentation, and academic portfolio purposes.
