# item_recommendation_system

## for json to csv

1. run itt in powershel

python scripts/json_to_csv.py --output data\convert_to_json_to_CSV.csv --flatten

3. then JSON paste 
4. then go to new line and type END 

## Cleaning dataset

1. paste this into powershel

python scripts/clean_sales_dataset.py --input "YOUR_DATASET.csv" --output "data\cleaned.csv" --report "data\clean_report.json" --min-item-freq 2


## Unique Item LIst

1. paste this into powershel

python scripts/extract_unique_item_names.py --input "data\cleaned.csv" --output-txt "data\unique_item_names.txt" --output-json "data\unique_item_names_report.json"

# feature builidng

1. python scripts/build_features.py --input data/form_paste_clean_v2.csv --output-dir data/features

# data one hot matrix
1. python scripts/build_features.py --input data/form_paste_clean_v2.csv --output-dir data/features --dense-onehot

                                **📊 Preprocessing Documentation

This document describes the full data preprocessing pipeline applied to the sales dataset. The goal is to clean, standardize, and structure the data for further analysis and recommendation systems.**
- Data cleaning & normalization
- Invalid data removal
- Outlier detection
- Item name standardization & merging
- Order id 
- basket construction
- Rare item filtering