# Multi-Turn / Interactive Text-to-SQL



## 1) Create & Activate a Virtual Environment

macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2) Install Dependencies
```
pip install -r requirements.txt
```


## 3) Directory Layout
```
 /
 ├─ data/
 │   ├─ dev.json              # BIRD dev set (questions + metadata)
 │   ├─ dev_databases/        # ← put the BIRD .sqlite databases here
 │   ├─ dev_tables.json       # extra metadata
 │   └─ dev_gold.sql          # gold SQL 
 ├─ exp_result/
 │   ├─ log/                  # logs are written here
 │   └─ result/               # batch predictions file goes here
 ├─ run/
 │   ├─ run_gpt.sh            # batch prediction
 │   ├─ run_evaluation.sh     # execution-based evaluation
 │   └─ run_interactive.sh    # interactive (human feedback) mode
 └─ src/
     ├─ gpt_request.py        # batch runner
     └─ interactive_request.py# human-in-the-loop runner 
```

**About data/dev_databases/**

- The folder data/dev_databases/ must contain the official BIRD database files.
- Each subdirectory corresponds to a database mentioned in dev.json, and should include a single .sqlite file containing all the tables used by that schema.
- Refer to [https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)

## 4) Configure Your OpenAI API Key
Open run/run_gpt.sh and run/run_interactive.sh, then set:
```
api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

## 5) How to Run

### A. Batch Prediction (entire dev set)

Generates SQL for all dev questions and logs the prompts + raw model outputs.

```bash
sh ./run/run_gpt.sh
```

**Outputs**

- Predictions: ./exp_result/result/predict_dev.json

- Prompt/Response log (human-readable): ./exp_result/log/prompt_log.txt

The predictions file stores one prediction per index, with the db_id appended (e.g., \t----- bird -----\t<db_id>). This format matches the downstream evaluator.


**Evaluation**

This runs both the predicted SQL and the gold SQL against the appropriate SQLite database and checks equality of result sets.

```
sh ./run/run_evaluation.sh
```

### B. Interactive Mode (human-in-the-loop)

Pick a single question by index, generate a first SQL, execute it, compare to gold by execution, and — if incorrect — add feedback iteratively. All prompts/feedback/SQL per step are logged.

```
sh ./run/run_interactive.sh
```


You will be prompted:

- Enter question index (0 ~ N-1):
- (each loop) Provide manual feedback? (y/n):

If you choose n, the script can generate simple automatic feedback (e.g., missing WHERE, JOIN, GROUP BY) to help the model refine the query.

**Outputs**

- Final interactive record: ./exp_result/interactive_result.json
- Full interactive transcript (prompts, feedback, SQL, execution result, per step):
./exp_result/log/interactive_log.txt