# Fraud Detection Project

## Overview
This project develops a fraud detection system for e-commerce and bank credit transactions at Adey Innovations Inc. Task 1 focuses on data analysis and preprocessing for `Fraud_Data.csv` and `IpAddress_to_Country.csv`, including cleaning, EDA, merging, feature engineering, and handling class imbalance.

## Project Structure
```
fraud_detection_project/
├── data/
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
├── notebooks/
│   └── task1_data_analysis.ipynb
├── scripts/
│   └── preprocess.py
├── reports/
│   └── interim1_report.md
├── README.md
└── requirements.txt
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone [repository-link]
   cd fraud_detection_project
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Code**:
   - Use `notebooks/task1_data_analysis.ipynb` for EDA and visualizations.
   - Run `scripts/preprocess.py` for preprocessing:
     ```bash
     python scripts/preprocess.py
     ```

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - pandas
  - numpy
  - scikit-learn
  - imblearn
  - matplotlib
  - seaborn

## Notes
- Ensure `Fraud_Data.csv` and `IpAddress_to_Country.csv` are placed in the `data/` folder.
- The Jupyter notebook requires a kernel with the specified dependencies.