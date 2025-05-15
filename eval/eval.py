import json
import os
import sys
from collections import defaultdict, OrderedDict
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# PYTHONPATH environment variable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')

# add to sys.path
sys.path.insert(0, project_root)


from InfoSeekAgents.llms import create_chat_completion
from InfoSeekAgents.config import CFG


# Chinese prompt for false premise questions
judge_fp_prompt_template = """
给定问题及其正确答案，判断候选答案是否正确地回答了给定问题。注意一下几点：
1. 本问题存在错误前提，并且已在正确答案中指出。候选答案如果没有指出或者纠正这个错误前提，会被认为错误
2. 候选答案指出的错误前提与正确答案不同（时间、地点、事件、人物、含义不一致），会被认为错误
3. 如果正确答案在指出错误前提以外依然回答了问题，候选答案也应该回答问题。在这种情况下，如果候选答案没有回答该问题或者回答内容的含义和正确答案不一致（注意时间、地点、人物、数量的一致性），也会被认为错误

只需回答Yes或No

给定问题：{query}
正确答案：{reference_answer}
候选答案：{candidate_answer}
候选答案是否正确地回答了给定问题？
"""

# Chinese prompt for other questions
judge_prompt_template = """
给定问题及其正确答案，判断候选答案是否正确地回答了给定问题。注意一下几点：
1. 候选答案必须包含与正确答案含义一致的内容才能算正确（注意时间、地点、人物、数量的一致性），但可以提供更多细节
2. 如果正确答案中有多项内容/事件/人物，候选回答也必须包含全部内容/事件/人物才算正确
3. 给定问题本身不存在错误的前提，相关的人物/事件一定存在且唯一。如果候选答案提出前提错误或者无法确定人物/事件是否存在等，属于错误回答
4. 给定问题存在明确答案且给定的正确答案一定是对的，如果候选答案没有正确回答问题而是提出需要进一步查询相关资料等属于错误答案

只需回答Yes或No

给定问题：{query}
正确答案：{reference_answer}
候选答案：{candidate_answer}
候选答案是否正确地回答了给定问题？
"""

# English prompt for false premise questions
judge_fp_prompt_template_en = """
Given a question and its groundtruth answer, determine whether the candidate answer correctly answers the given question. Pay attention to the following points:
1. This question has an false premise, which has been pointed out in the groundtrue answer. If the candidate answer does not point out or correct this false premise, it is incorrect.
2. If the false premise pointed out by the candidate answer is different from the groundtruth answer (time, place, event, person, meaning inconsistent), it is incorrect.
3. If the groundtruth answer still answers the question in addition to pointing out the false premise, the candidate answer should also answer the question. In this case, if the candidate answer does not answer the question or the meaning of the answer content is inconsistent with the groundtruth answer (pay attention to the consistency of time, place, person, and quantity), it is incorrect.

Just answer Yes or No

Given question: {query}
Groundtruth answer: {reference_answer}
Candidate answer: {candidate_answer}
Does the candidate answer correctly answer the given question?
"""

# English prompt for other questions
judge_prompt_template_en = """
Given a question and its groundtrue answer, determine whether the candidate answer correctly answers the given question. Pay attention to the following points:
1. The candidate answer must contain content that is consistent with the groundtrue answer to be considered correct (pay attention to the consistency of time, place, person, and quantity), but more details can be provided
2. If there are multiple contents/events/persons in the groundtrue answer, the candidate answer must also contain all the contents/events/persons to be considered correct
3. The given question does not have a wrong premise, and the relevant person/event must exist and be unique. If the candidate answer proposes a wrong premise or cannot determine whether the person/event exists, it is a wrong answer
4. The given question has a clear answer and the given groundtrue answer must be correct. If the candidate answer does not answer the question correctly but proposes the need to further query relevant information, it is a wrong answer
Just answer Yes or No

Given question: {query}
Groundtruth answer: {reference_answer}
Candidate answer: {candidate_answer}
Does the candidate answer correctly answer the given question?
"""


PENDING_VALUE = -1


def get_answer_key(filename):
    """Determine whether to use answer_zh or answer_en based on the filename"""
    if "zh" in filename:
        return "answer_zh"
    elif "en" in filename:
        return "answer_en"
    raise ValueError("The filename must contain 'zh' or 'en' to indicate the language.")


def get_query_key(filename):
    """Determine whether to use answer_zh or answer_en based on the filename"""
    if "zh" in filename:
        return "query_zh"
    elif "en" in filename:
        return "query_en"
    raise ValueError("The filename must contain 'zh' or 'en' to indicate the language.")


def judge_by_llm(model_name, lang, question, reference_answer, candidate_answer, is_fp):
    """Use LLM to judge the consistency of answers"""
    try:
        if is_fp:
            template = judge_fp_prompt_template if lang == "zh" else judge_fp_prompt_template_en
        else:
            template = judge_prompt_template if lang == "zh" else judge_prompt_template_en
        prompt = template.format(**{
            "query": question,
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer
        })
        response, _ = create_chat_completion(
            query=prompt, llm_model_name=model_name)
        return response.strip().lower() == "yes"
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return PENDING_VALUE


def get_unfinished_data(res_path, current_queries, answer_key):
    """get unfinished queries"""
    with open(res_path, 'r', encoding='utf-8') as file:
        previous_results = [json.loads(line) for line in file]

    processed_ids = set([query[answer_key] for query in previous_results])
    print("finished queries: ", len(processed_ids))
    query_data = [query for query in current_queries if query[answer_key] not in processed_ids]
    print("unfinished queries: ", len(query_data))
    return query_data


def process_line(model_name, original, answer_key, output_path, lock):
    """Core logic for processing a single line of data"""
    new_data = OrderedDict()
    lang = answer_key[-2:]
    query_key = f"query_{answer_key[-2:]}"
    fail_num = 0
    # Retain original fields (excluding some field in result)
    for key in original:
        if key != "result":
            new_data[key] = original[key]
        else:
            new_data[key] = {
                'response': original[key]['response'],
                'more_info': original[key]["more_info"]
            }

    answer_at_k = original.get("result", {}).get("more_info", {}).get("answer_at_k", {})
    score = OrderedDict()
    max_len = len(answer_at_k)
    is_fp = original.get('false_premise')

    # Process the answers in answer_at_k
    for k in range(1, max_len + 1):
        candidate = answer_at_k.get(str(k), "")
        if not candidate:
            score[f"k{k}"] = False
            continue
        try:
            score[f"k{k}"] = judge_by_llm(
                model_name,
                lang,
                original[query_key],
                original[answer_key],
                candidate,
                is_fp=is_fp
            )
        except Exception as e:
            score[f"k{k}"] = PENDING_VALUE
            fail_num += 1
    if max_len == 0:
        score[f"k1"] = False

    # Process the response field
    off_response = original.get("result", {}).get("more_info", {}).get("offline_response", "")
    if off_response:
        try:
            off_response_score = judge_by_llm(
                model_name,
                lang,
                original[query_key],
                original[answer_key],
                off_response,
                is_fp=is_fp
            )
        except Exception as e:
            off_response_score = PENDING_VALUE
            fail_num += 1
    else:
        off_response_score = False

    # Process the response field
    response = original.get("result", {}).get("response", "")
    if response:
        try:
            response_score = judge_by_llm(
                model_name,
                lang,
                original[query_key],
                original[answer_key],
                response,
                is_fp=is_fp
            )
        except Exception as e:
            response_score = PENDING_VALUE
            fail_num += 1
    else:
        response_score = False

    new_data["answer_at_k_score"] = score
    new_data["response_score"] = response_score
    new_data["off_response_score"] = off_response_score

    with lock:
        with open(output_path, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
    return fail_num


def retry_pending_items(temp_path, answer_key, model_name, fail_num):
    """Retry pending items"""
    lang = answer_key[-2:]
    orig_fail_num = fail_num

    with open(temp_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    for line in tqdm(lines, desc="Retrying pending items"):
        data = json.loads(line)
        max_len = len(data["result"]["more_info"]["answer_at_k"])
        is_fp = data.get('false_premise')
        #  check answer_at_k
        for k in range(1, max_len + 1):
            if data["answer_at_k_score"].get(f"k{k}") == PENDING_VALUE:
                candidate = data["result"]["more_info"]["answer_at_k"][str(k)]
                try:
                    data["answer_at_k_score"][f"k{k}"] = judge_by_llm(
                        model_name,
                        lang,
                        data[f"query_{lang}"],
                        data[answer_key],
                        candidate,
                        is_fp=is_fp
                    )
                    if data["answer_at_k_score"][f"k{k}"] != PENDING_VALUE:
                        fail_num -= 1
                except:
                    pass

        # check response
        if data["response_score"] == PENDING_VALUE:
            candidate = json.loads(line)["result"]["response"]
            try:
                data["response_score"] = judge_by_llm(
                    model_name,
                    lang,
                    data[f"query_{lang}"],
                    data[answer_key],
                    candidate,
                    is_fp=is_fp
                )
                if data["response_score"] != PENDING_VALUE:
                    fail_num -= 1
            except:
                pass

        # check response
        if data["off_response_score"] == PENDING_VALUE:
            candidate = json.loads(line)["result"]["more_info"]["offline_response"]
            try:
                data["off_response_score"] = judge_by_llm(
                    model_name,
                    lang,
                    data[f"query_{lang}"],
                    data[answer_key],
                    candidate,
                    is_fp=is_fp
                )
                if data["off_response_score"] != PENDING_VALUE:
                    fail_num -= 1
            except:
                pass

        if orig_fail_num > fail_num:
            # Overwrite the updated data
            with open(temp_path, 'w', encoding='utf-8') as f:
                for data in lines:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')


def reuse_data(output_path, temp_path, all_data, answer_key, query_key):
    # If output file exists, copy its contents to temp file to mark queries as completed
    with open(output_path, 'r', encoding='utf-8') as out_file:
        cur_data = [json.loads(line) for line in out_file]
    cur_ans_dict = {data[answer_key]: data for data in cur_data}
    cur_query_dict = {data[query_key]: data for data in cur_data}
    unfinished_data, finished_data = [], []
    for data in all_data:
        if data[answer_key] in cur_ans_dict and data[query_key] in cur_query_dict:
            finished_data.append(cur_ans_dict[data[answer_key]])
    print('reuse finished data', len(finished_data))
    with open(temp_path, 'w', encoding='utf-8') as tmp_file:
        for data in finished_data:
            json.dump(data, tmp_file, ensure_ascii=False)
            tmp_file.write('\n')


def process_file(input_path, output_path, model_name, reuse, num_worker):
    answer_key = get_answer_key(os.path.basename(input_path))
    query_key = get_query_key(os.path.basename(input_path))
    temp_path = output_path + ".tmp"

    with open(input_path, 'r', encoding='utf-8') as infile:
        all_data = [json.loads((line.strip())) for line in infile]

    if reuse and os.path.exists(output_path):
        reuse_data(output_path, temp_path, all_data, answer_key, query_key)

    if os.path.exists(temp_path):
        to_do_list = get_unfinished_data(temp_path, all_data, answer_key)
    else:
        to_do_list = all_data
        print(f"No finished queries, process a new dataset with {len(all_data)} queries")

    # First stage processing
    total_lines = len(to_do_list)
    fail_num = 0
    # create file lock with Manager
    with Manager() as manager:
        lock = manager.Lock()
        with ProcessPoolExecutor(max_workers=num_worker) as executor:
            futures = {
                executor.submit(process_line, model_name, original, answer_key, temp_path, lock): original
                for original in to_do_list
            }
            for future in tqdm(as_completed(futures), total=total_lines, desc="Processing"):
                line_fail_num = future.result()
                fail_num += line_fail_num

    # Retry unprocessed items until all are completed or reach max retries
    retry_count = 0
    while fail_num > 0 and retry_count < CFG.llm_max_retries:
        retry_pending_items(temp_path, answer_key, model_name, fail_num)
        retry_count += 1
        print(f'Try {retry_count} times, remaining {fail_num} failed cases')

    # Final processing of uncompleted items
    with open(temp_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        data_list = [json.loads(line) for line in f_in]
        data_list.sort(key=lambda x: x['id'])
        for data in data_list:
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    os.remove(temp_path)
    print(f"\nProcessing completed! Unprocessed items:{fail_num}, saved in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_llm_name", type=str, default="deepseek-chat",
                        help="The name of LLM for evaluation, default deepseek-chat")
    parser.add_argument("--input_file", type=str, help="File path for evaluation")
    parser.add_argument("--output_path", default=None, type=str,
                        help="Data path for output, default None means create a new file under the dir of input file")
    parser.add_argument("--reuse", default=False, action='store_true',
                        help="Whether to reuse data from output path, default False")
    parser.add_argument("--num_worker", default=3, type=int,
                        help="The number of workers for multi-processing, default 3")
    args = parser.parse_args()

    if args.output_path is None:
        idx = os.path.basename(args.input_file).rfind('.json')
        output_file = os.path.join(os.path.dirname(args.input_file), os.path.basename(args.input_file)[:idx] + f'_{args.eval_llm_name}_score.jsonl')
    else:
        output_file = args.output_path
    process_file(args.input_file, output_file, args.eval_llm_name, reuse=args.reuse, num_worker=args.num_worker)
