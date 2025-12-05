#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import backoff
from openai import OpenAI, APIError, RateLimitError, APITimeoutError



def init_client(api_key: str = None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAI(api_key=api_key)
    else:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("❌ No API key provided. Use --api_key or set OPENAI_API_KEY.")
        return OpenAI(api_key=key)



def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def quote_identifier(name: str) -> str:
    """Safely quote SQLite identifiers."""
    return '"' + name.replace('"', '""') + '"'


def load_column_descriptions(db_path: str) -> Dict[str, Dict[str, str]]:
    """Load column descriptions from optional CSV files under database_description."""
    db_dir = os.path.dirname(db_path)
    description_dir = os.path.join(db_dir, "database_description")
    descriptions: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(description_dir):
        return descriptions

    for file_name in os.listdir(description_dir):
        if not file_name.lower().endswith(".csv"):
            continue
        table_name = os.path.splitext(file_name)[0].lower()
        file_path = os.path.join(description_dir, file_name)
        table_desc: Dict[str, str] = {}
        decoded = False
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding, newline="") as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        description = (row.get("column_description") or "").strip()
                        col_candidates = [
                            (row.get("original_column_name") or "").strip(),
                            (row.get("column_name") or "").strip()
                        ]
                        for col_name in col_candidates:
                            if not col_name:
                                continue
                            table_desc[col_name.lower()] = description
                decoded = True
                break
            except UnicodeDecodeError:
                continue
            except (OSError, csv.Error):
                table_desc = {}
                break
        if decoded and table_desc:
            descriptions[table_name] = table_desc
    return descriptions


def fetch_column_examples(conn: sqlite3.Connection, table_name: str, column_name: str, limit: int = 3) -> List[str]:
    """Fetch up to `limit` distinct non-null example values for a column."""
    cursor = conn.cursor()
    tbl = quote_identifier(table_name)
    col = quote_identifier(column_name)
    try:
        cursor.execute(f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL LIMIT ?", (limit,))
        values = cursor.fetchall()
    except sqlite3.Error:
        return []

    examples = []
    for (value,) in values:
        if value is None:
            continue
        examples.append(str(value))
    return examples


def get_foreign_keys(conn: sqlite3.Connection, table_name: str) -> Dict[str, Tuple[str, str]]:
    """Return mapping from column -> (referenced_table, referenced_column)."""
    cursor = conn.cursor()
    try:
        cursor.execute(f'PRAGMA foreign_key_list({quote_identifier(table_name)})')
        fk_info = cursor.fetchall()
    except sqlite3.Error:
        return {}

    fk_map: Dict[str, Tuple[str, str]] = {}
    for row in fk_info:
        # row schema: (id, seq, table, from, to, on_update, on_delete, match)
        if len(row) >= 5:
            from_col = row[3]
            ref_table = row[2]
            ref_column = row[4]
            fk_map[from_col] = (ref_table, ref_column)
    return fk_map


def build_table_prompt(
    conn: sqlite3.Connection,
    table_name: str,
    descriptions: Dict[str, Dict[str, str]],
    sample_limit: int,
    fk_relations: List[str],
) -> str:
    cursor = conn.cursor()
    quoted_table = quote_identifier(table_name)
    try:
        cursor.execute(f"PRAGMA table_info({quoted_table})")
        columns_info = cursor.fetchall()
    except sqlite3.Error:
        return ""

    if not columns_info:
        return ""

    fk_map = get_foreign_keys(conn, table_name)
    table_desc_map = descriptions.get(table_name.lower(), {})

    column_lines = []
    for idx, col in enumerate(columns_info):
        # col schema: (cid, name, type, notnull, dflt_value, pk)
        col_name = col[1]
        col_type = col[2] or "UNKNOWN"
        is_pk = bool(col[5])
        description = table_desc_map.get(col_name.lower(), "").strip()
        if description:
            description = " ".join(description.split())
        fk_info = fk_map.get(col_name)
        if fk_info:
            fk_relations.append(f"{table_name}.{col_name} = {fk_info[0]}.{fk_info[1]}")

        parts = [f"{col_name}: {col_type}"]
        if is_pk:
            parts.append("Primary Key")
        if description and fk_info:
            lower_desc = description.lower()
            maps_idx = lower_desc.find("maps to")
            if maps_idx != -1:
                description = description[:maps_idx].rstrip(", ")
        if description:
            parts.append(description)

        parts_text = ", ".join(parts)
        examples = fetch_column_examples(conn, table_name, col_name, limit=sample_limit)
        examples_str = ", ".join(examples[:sample_limit]) if examples else ""
        line = f"  ({parts_text}"
        if fk_info:
            line += f"\n   Maps to {fk_info[0]}({fk_info[1]})"
        if examples_str:
            line += f", Examples: [{examples_str}]"
        line += ")"
        if idx < len(columns_info) - 1:
            line += ","
        column_lines.append(line)

    table_block = f"# Table: {table_name}\n[\n" + "\n\n".join(column_lines) + "\n]"
    return table_block


def generate_schema_prompt(db_path: str, sample_limit: int = 3):
    """Build an m-schema style prompt with column metadata, examples, and foreign keys."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = sorted(row[0] for row in cursor.fetchall())

    descriptions = load_column_descriptions(db_path)
    schema_sections = []
    fk_relations: List[str] = []

    for table_name in tables:
        table_prompt = build_table_prompt(conn, table_name, descriptions, sample_limit, fk_relations)
        if table_prompt:
            schema_sections.append(table_prompt)

    conn.close()

    if fk_relations:
        unique_fk = sorted(set(fk_relations))
        fk_section = "【Foreign keys】\n" + "\n".join(unique_fk)
        schema_sections.append(fk_section)

    return "\n\n".join(schema_sections)


def generate_comment_prompt(question, knowledge=None):
    pattern_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_kg = "-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above. Return only the SQL query. Do not provide any explanation."
    question_prompt = f"-- {question}"
    if knowledge:
        knowledge_prompt = f"-- External Knowledge: {knowledge}"
        return knowledge_prompt + '\n' + pattern_kg + '\n' + question_prompt
    else:
        return pattern_no_kg + '\n' + question_prompt


# def cot_wizard():
#     return "\nGenerate the SQL only after thinking step by step: "


def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path)
    comment_prompt = generate_comment_prompt(question, knowledge)
    combined_prompts = schema_prompt + '\n\n' + comment_prompt + '\nSELECT '
    return combined_prompts



def quota_giveup(e):
    return isinstance(e, RateLimitError)


@backoff.on_exception(
    backoff.constant,
    (APIError, RateLimitError, APITimeoutError, Exception),
    giveup=quota_giveup,
    interval=15,
    max_tries=3
)
def connect_gpt(client, engine, prompt, max_tokens=256, temperature=0, stop=None):
    """Call GPT and return text output."""
    result = client.completions.create(
        model=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or [";", "#", "--"]
    )
    return result.choices[0].text



def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None, log_dir="./exp_result/log/"):
    client = init_client(api_key)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "prompt_log.txt")

    responses = []
    with open(log_path, "a", encoding="utf-8") as log_f:
        for i, question in tqdm(enumerate(question_list), total=len(question_list)):
            db_id = os.path.basename(db_path_list[i]).replace('.sqlite', '')
            print(f"--------------------- processing {i}th question ({db_id}) ---------------------")
            print(f"Question: {question}")

            if knowledge_list:
                prompt = generate_combined_prompts_one(db_path=db_path_list[i], question=question, knowledge=knowledge_list[i])
            else:
                prompt = generate_combined_prompts_one(db_path=db_path_list[i], question=question)

            try:
                sql_text = connect_gpt(client, engine=engine, prompt=prompt)
            except Exception as e:
                sql_text = f"error:{e}"

            sql = 'SELECT' + sql_text if not sql_text.strip().upper().startswith('SELECT') else sql_text
            sql = sql.strip() + f"\t----- bird -----\t{db_id}"
            responses.append(sql)

            log_f.write("============================================================\n")
            log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  Q#{i}  DB: {db_id}\n")
            log_f.write(f"Question: {question}\n")
            log_f.write(f"\n----- Prompt Sent to {engine} -----\n{prompt}\n")
            log_f.write(f"\n----- LLM Response -----\n{sql_text.strip()}\n")
            log_f.write("============================================================\n\n")
            log_f.flush()  # 매 쿼리마다 바로 기록됨

    print(f"Prompt log saved to: {log_path}")
    return responses



def decouple_question_schema(datasets, db_root_path):
    question_list, db_path_list, knowledge_list = [], [], []
    for data in datasets:
        question_list.append(data['question'])
        cur_db_path = os.path.join(db_root_path, data['db_id'], f"{data['db_id']}.sqlite")
        db_path_list.append(cur_db_path)
        knowledge_list.append(data.get('evidence', None))
    return question_list, db_path_list, knowledge_list


def generate_sql_file(sql_lst, output_path=None):
    result = {i: sql for i, sql in enumerate(sql_lst)}
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        json.dump(result, open(output_path, 'w'), indent=4)
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--use_knowledge', type=str, default='False')
    parser.add_argument('--db_root_path', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--engine', type=str, default='gpt-4o-mini')
    parser.add_argument('--data_output_path', type=str, default='./exp_result/')
    parser.add_argument('--chain_of_thought', type=str, default='False')
    parser.add_argument('--log_dir', type=str, default='./exp_result/log/')
    args = parser.parse_args()

    eval_data = json.load(open(args.eval_path))
    question_list, db_path_list, knowledge_list = decouple_question_schema(eval_data, args.db_root_path)

    if args.use_knowledge == 'True':
        responses = collect_response_from_gpt(
            db_path_list, question_list, args.api_key, args.engine,
            knowledge_list, log_dir=args.log_dir
        )
    else:
        responses = collect_response_from_gpt(
            db_path_list, question_list, args.api_key, args.engine,
            knowledge_list=None, log_dir=args.log_dir
        )

    if args.chain_of_thought == 'True':
        output_name = os.path.join(args.data_output_path, f"predict_{args.mode}_cot.json")
    else:
        output_name = os.path.join(args.data_output_path, f"predict_{args.mode}.json")

    generate_sql_file(sql_lst=responses, output_path=output_name)
    print(f"Successfully collected results from {args.engine} for {args.mode}; ")
