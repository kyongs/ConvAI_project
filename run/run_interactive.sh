#!/bin/bash
eval_path="./data/dev.json"
gold_path="./data/dev_gold.sql"
db_root_path="./data/dev_databases/"
api_key=''
output_path="./exp_result/interactive_result.json"

echo "=== Starting Interactive Text-to-SQL ==="
python3 -u ./src/interactive_request.py \
    --eval_path ${eval_path} \
    --gold_path ${gold_path} \
    --db_root_path ${db_root_path} \
    --api_key ${api_key} \
    --output_path ${output_path} \
    --max_iter 5
