# Titanic SQL + EDA Project

This project explores the Titanic dataset using both SQL queries (via SQLite) and Python-based EDA using pandas, matplotlib, and seaborn.

## Features
- Load Titanic CSV data
- Store in in-memory SQLite database
- Run SQL query to calculate survival rate by class and sex

## File Structure
- `data/titanic.csv` – Dataset file
- `scripts/titanic_sql_analysis.py` – Script with SQL + Pandas workflow
- `notebooks/titanic_sql_eda.ipynb` – Same analysis in Jupyter Notebook
- `output/` – Folder to save results (e.g., CSVs, charts)

## How to Run
1. Place `titanic.csv` in the `data/` folder.
2. Open the notebook in `notebooks/` and run each cell.
3. Use the script in `scripts/` for CLI-based analysis.

## Requirements
- pandas
- matplotlib
- seaborn
- sqlite3 (built-in)

## Dependencies
```bash
pip install pandas
