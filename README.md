# Student Performance Analytics System

This project analyzes student academic data to identify factors influencing final performance.

## Features

- Web application with dataset upload (CSV/XLSX)
- Descriptive statistics (mean, median, variance, standard deviation, min, max)
- Interactive correlation heatmap and matrix for numeric variables
- Factor importance ranking (absolute correlation with target)
- Interactive visualizations:
  - Histograms
  - Box plots
  - Scatter plots with OLS regression lines vs detected final-performance target
- OLS regression summary (R-squared, coefficients, p-values)
- Data preview table
- Also includes CLI script for file-based analysis

## Dataset

The default dataset is `StudentsPerformance.csv` (placed in project root). It contains:
- gender, race/ethnicity, parental education, lunch, test preparation
- math score, reading score, writing score

The app auto-computes `Final_Grade` as the mean of available score columns if not present. You can also upload your own dataset with columns like `Attendance`, `Internal_Marks`, `Assignment_Scores`, `Study_Hours`, and `Final_Grade`.

## Project Structure

```text
.
├── .gitignore
├── app.py
├── README.md
├── report.pdf
├── requirements.txt
├── StudentsPerformance.csv
├── student_performance_analysis.py
├── TODO.md
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── data/                # Optional for CLI mode (create if needed)
└── output/              # Auto-created in CLI mode
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Web App (Recommended)

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Upload your dataset from the browser and view results directly.

## LAN Hosting (Same Wi-Fi Network)

Run the app with host `0.0.0.0` so other devices in your local network can access it:

```bash
python app.py --host 0.0.0.0 --port 5000
```

Then open on another device using your PC's LAN IP:

```text
http://YOUR_LOCAL_IP:5000
```

Example:

```text
http://192.168.1.10:5000
```

Optional debug mode:

```bash
python app.py --host 0.0.0.0 --port 5000 --debug
```

If access is blocked, allow Python/port `5000` through Windows Firewall.

## Run CLI Script (Optional)

```bash
python student_performance_analysis.py --file StudentsPerformance.csv
```

If `--file` is omitted, the script picks the first CSV/XLSX in `data/`.

## CLI Outputs

Inside `output/`, you will get:

- `descriptive_statistics.csv`
- `correlation_matrix.csv`
- `correlation_heatmap.png`
- `histograms.png`
- `boxplots.png`
- `scatterplots_vs_target.png` (if target column found)

## Notes

- The system auto-detects target columns like `Final Grade`, `Final Marks`, `GPA`, or any `*score*` column.
- Categorical columns are excluded from numeric statistics unless encoded.
