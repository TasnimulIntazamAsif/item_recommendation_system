# item_recommendation_system

## for json to csv

1. run itt in powershel
2. 
python scripts/json_to_csv.py --output data\convert_to_json_to_CSV.csv -- flatten

3. then JSON paste 
4. then go to new line and type END 

## Cleaning dataset

1. paste this into powershel

python scripts/clean_sales_dataset.py --input "YOUR_DATASET.csv" --output "data\cleaned.csv" --report "data\clean_report.json" --min-item-freq 2


## Unique Item LIst

1. paste this into powershel

python scripts/extract_unique_item_names.py --input "data\cleaned.csv" --output-txt "data\unique_item_names.txt" --output-json "data\unique_item_names_report.json"