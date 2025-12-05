#!/usr/bin/env python3
import os, json, argparse, sqlite3
from datetime import datetime
from openai import OpenAI
from gpt_request import generate_schema_prompt, generate_comment_prompt
from tqdm import tqdm
import re



def execute_and_compare(predicted_sql, gold_sql, db_path):
    """
    두 SQL을 실제 DB에 실행시켜 결과를 비교함.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(predicted_sql)
        pred_res = cursor.fetchall()
    except Exception as e:
        conn.close()
        return False, f"Predicted SQL Execution Error: {e}"

    try:
        cursor.execute(gold_sql)
        gold_res = cursor.fetchall()
    except Exception as e:
        conn.close()
        return False, f"Gold SQL Execution Error: {e}"

    conn.close()
    if set(pred_res) == set(gold_res):
        return True, "Execution results match "
    else:
        return False, "Execution results differ "


# ============================================================
# Feedback 자동 생성
# ============================================================
def auto_feedback(pred_sql, gold_sql):
    """단순 rule-based feedback"""
    if "where" not in pred_sql.lower() and "where" in gold_sql.lower():
        return "You forgot the WHERE condition."
    elif "group by" in gold_sql.lower() and "group by" not in pred_sql.lower():
        return "You missed the GROUP BY clause."
    elif "join" in gold_sql.lower() and "join" not in pred_sql.lower():
        return "You should include a JOIN operation."
    else:
        return "The SQL result is incorrect. Please refine conditions or joins."



def call_llm(client, prompt, model="gpt-4o-mini", temperature=0.1):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Text-to-SQL expert. Output only valid SQL code."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()



def build_interactive_prompt(schema_prompt, question, feedback_history, pred_sql):
    base_instruction = generate_comment_prompt(question, knowledge=None) + "\n-- Return only the SQL query."
    if not pred_sql:
        return f"{schema_prompt}\n\n{base_instruction}"

    feedback_str = "\n".join([f"- {fb}" for fb in feedback_history]) if feedback_history else "- (none)"
    return f"""{schema_prompt}

-- Previous SQL (incorrect):
{pred_sql}

-- User feedback:
{feedback_str}

-- Refine the SQL query accordingly.

{base_instruction}
"""


def interactive_loop(client, question, db_path, gold_sql, max_iter=3, model="gpt-4o-mini",
                     log_path="./exp_result/log/interactive_log.txt"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    schema_prompt = generate_schema_prompt(db_path)
    feedback_history = []
    pred_sql = ""
    log_f = open(log_path, "a", encoding="utf-8")

    for step in range(max_iter):
        # 프롬프트 생성
        prompt = build_interactive_prompt(schema_prompt, question, feedback_history, pred_sql if step else "")

        # LLM 호출
        pred_sql = call_llm(client, prompt, model=model)
        pred_sql = re.sub(r"```sql|```", "", pred_sql, flags=re.IGNORECASE).strip()

        # 실행 및 결과 비교
        same, message = execute_and_compare(pred_sql, gold_sql, db_path)
        print(f"\n[Step {step+1}] Predicted SQL:\n{pred_sql}")
        print(f"→ Execution Check: {message}")

        # 로그 작성
        log_f.write("============================================================\n")
        log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step {step+1}\n")
        log_f.write(f"Question: {question}\n")
        log_f.write(f"Prompt:\n{prompt}\n")
        log_f.write(f"Predicted SQL:\n{pred_sql}\n")
        log_f.write(f"Execution Result: {message}\n")

        if same:
            print("Correct SQL found (Execution results match)!\n")
            log_f.write("Result: Correct SQL (execution match)\n")
            break

        # 사용자 피드백 입력
        feedback_mode = input("Provide manual feedback? (y/n): ").strip().lower()
        if feedback_mode == "y":
            feedback = input("Enter feedback: ").strip()
        else:
            feedback = auto_feedback(pred_sql, gold_sql)
            print("Auto Feedback:", feedback)

        feedback_history.append(feedback)
        log_f.write(f"Feedback added: {feedback}\n")

    log_f.write("============================================================\n\n")
    log_f.close()
    return {
        "question": question,
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "execution_result": message,
        "feedback_history": feedback_history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--gold_path", type=str, required=True)
    parser.add_argument("--db_root_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./exp_result/interactive_results.json")
    parser.add_argument("--max_iter", type=int, default=3)
    args = parser.parse_args()

    # 데이터 로드
    data = json.load(open(args.eval_path))
    gold_lines = [line.strip().split("\t")[0] for line in open(args.gold_path, "r")]
    client = OpenAI(api_key=args.api_key)

    # 사용자 입력
    idx = int(input(f"Enter question index (0 ~ {len(data)-1}): "))
    item = data[idx]

    db_path = os.path.join(args.db_root_path, item["db_id"], f"{item['db_id']}.sqlite")
    question = item["question"]
    gold_sql = gold_lines[idx] 

    print(f"\nSelected Question #{idx}: {question}")
    print(f"Gold SQL: {gold_sql}")

    result = interactive_loop(
        client,
        question,
        db_path,
        gold_sql,
        max_iter=args.max_iter,
        model="gpt-4o-mini",
        log_path="./exp_result/log/interactive_log.txt"
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nInteractive session finished. Result saved to {args.output_path}")
