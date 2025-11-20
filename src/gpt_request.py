#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict
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


def generate_schema_prompt(db_path, num_rows=None):
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table[0],))
        create_prompt = cursor.fetchone()[0]
        full_schema_prompt_list.append(create_prompt)

    return "\n\n".join(full_schema_prompt_list)



def generate_comment_prompt(question, knowledge=None):
    pattern_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_kg = "-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above. Return only the SQL query. Do not provide any explanation."
    question_prompt = f"-- {question}"
    if knowledge:
        knowledge_prompt = f"-- External Knowledge: {knowledge}"
        return knowledge_prompt + '\n' + pattern_kg + '\n' + question_prompt
    else:
        return pattern_no_kg + '\n' + question_prompt


def cot_wizard():
    return "\nGenerate the SQL after thinking step by step: "


def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path)
    comment_prompt = generate_comment_prompt(question, knowledge)
    combined_prompts = schema_prompt + '\n\n' + comment_prompt + cot_wizard() + '\nSELECT '
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
