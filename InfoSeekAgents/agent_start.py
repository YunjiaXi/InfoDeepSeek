import argparse
from datetime import datetime
import json
import os
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import Manager

from InfoSeekAgents.config import Config, CFG
from InfoSeekAgents.agents import InfoSeekAgent, AgentProfile


class AgentService(object):
    def __init__(self, *args, **kwargs):
        self.cfg = Config()
        self.agent_profile = None
        self.p_date = datetime.today().strftime('%Y%m%d')

    @staticmethod
    def parse_config(input_dict):
        cfg = Config()

        llm_name = input_dict.get("llm_name", "").lower()
        fast_llm_name = input_dict.get("fast_llm_name", "").lower()
        if not fast_llm_name:
            fast_llm_name = llm_name
        cfg.fast_llm_model = fast_llm_name
        cfg.smart_llm_model = llm_name
        cfg.search_type = input_dict.get("search_type", 'ddg')
        cfg.max_tokens_num = input_dict.get("max_tokens_num", 4096)
        cfg.max_search_nums = input_dict.get("max_search_nums", 5)
        cfg.print_to_console = input_dict.get("print_to_console", False)
        cfg.wo_tool = input_dict.get("wo_tool", False)
        cfg.lang_aware = input_dict.get('lang_aware', False)

        return cfg

    @staticmethod
    def load_history(input_dict):
        history = input_dict.get("history", list())
        if not history:
            history = list()
        if isinstance(history, str):
            history = json.loads(history)
        return history

    def chat(self, input_dict):
        s = "============ INPUT_DICT ============\n"
        for key, val in input_dict.items():
            s += f"· {key.upper()}:\t{val}\n"
        # print(s)

        chat_id = str(input_dict["id"])
        history = self.load_history(input_dict)
        self.cfg = self.parse_config(input_dict)
        self.agent_profile = AgentProfile(input_dict)

        try:
            agent = InfoSeekAgent(
                    cfg=self.cfg,
                    session_id=chat_id,
                    agent_profile=self.agent_profile,
                    lang=input_dict.get("lang", "en"))

            print("\033[95m\033[1m" + "\n***** Question *****" + "\033[0m\033[0m")
            print(input_dict["query"])

            agent_results = agent.chat(
                input_dict["query"], 
                history=history)

            print("\033[95m\033[1m" + "\n***** Response *****" + "\033[0m\033[0m")
            print(agent_results["response"])

            result = {
                "id": chat_id,
                "response": agent_results["response"],
                "more_info": agent_results["more_info"],
                "history": json.dumps(agent_results["history"], ensure_ascii=False),
                "full_llm_prompt_responses": agent_results["full_llm_prompt_responses"]
            }

        except KeyboardInterrupt:
            exit()
        except:
            print(traceback.format_exc())
            result = {
                "id": chat_id,
                "response": "error"
            }

        return result


def load_queries(query_path):
    with open(query_path, 'r', encoding='utf-8') as file:
        query_data = json.load(file)
    return query_data


def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False)


def get_unfinished_data(res_path, current_queries, query_key):
    """get unfinished queries"""
    with open(res_path, 'r', encoding='utf-8', errors='replace') as file:
        previous_results = [json.loads(line) for line in file]

    processed_ids = set([query[query_key] for query in previous_results])
    print("Current finished queries: ", len(processed_ids))
    query_data = [query for query in current_queries if query[query_key] not in processed_ids]
    print("Unfinished queries: ", len(query_data))
    return query_data


def get_unfinished_data_and_overwrite(res_path, current_queries, query_key):
    """get finished queries and overwrite"""
    previous_results = []
    with open(res_path, 'r', encoding='utf-8', errors='replace') as file:
        for i, line in enumerate(file):
            previous_results.append(json.loads(line))

    processed_ids = {query[query_key]: query for query in previous_results}
    cur_query_dict = {query[query_key]: query for query in current_queries}
    print("Current finished queries: ", len(processed_ids))
    exist_queries = []
    for query, value in processed_ids.items():
        if query in cur_query_dict:
            new_value = deepcopy(value)
            new_value['answer_zh'] = cur_query_dict[query]['answer_zh']
            new_value['answer_en'] = cur_query_dict[query]['answer_en']
            new_value['query_en'] = cur_query_dict[query]['query_en']
            new_value['query_zh'] = cur_query_dict[query]['query_zh']
            exist_queries.append(new_value)

    print("Existing finished queries: ", len(exist_queries), 'overwriting...')
    with open(res_path, 'w', encoding='utf-8') as file:
        for query in exist_queries:
            json.dump(query, file, ensure_ascii=False)
            file.write('\n')
    query_data = [query for query in current_queries if query[query_key] not in processed_ids]
    print("Unfinished queries: ", len(query_data))
    return query_data


def process_query(query, args, output_path, lock):
    """process each query"""
    try:
        agent_service = AgentService()
        args.query = query['query_en'] if args.lang == 'en' else query['query_zh']
        result = agent_service.chat(vars(args))
        query['result'] = result
        if result['response'] != 'error' and len(result["more_info"]) > 0:
            with lock:
                with open(output_path, 'a', encoding='utf-8') as file:
                    json.dump(query, file, ensure_ascii=False)
                    file.write('\n')
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print(f"Error processing query {query['id']}: {e}")
    return query


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="test", help="ID of this conversation")
    parser.add_argument("--query_path", type=str, help="Data path for user query")
    parser.add_argument("--output_path", default=None, type=str, help="Data path for output")
    parser.add_argument("--query", type=str, help="User query")
    parser.add_argument("--history", type=str, default='[]', help="History of conversation")
    parser.add_argument("--llm_name", type=str, default="gpt-4o", help="the name of llm")
    parser.add_argument("--fast_llm_name", type=str, default="", help="the name of fast llm")
    parser.add_argument("--use_local_llm", default=False, action='store_true', help="Whether to use local llm")
    parser.add_argument("--local_llm_host", type=str, default="localhost", help="The host of local llm service")
    parser.add_argument("--local_llm_port", type=int, default="8888", help="The port of local llm service")
    parser.add_argument("--print_to_console", default=False, action='store_true',
                        help="Whether display inner content in console, default False")

    parser.add_argument("--tool_names", type=str, default='["auto"]',
                        help='the name of tool set, ["auto"] for all tools, ["notool"] for no tool')
    parser.add_argument("--max_iter_num", type=int, default=5,
                        help="Max action step in retrieval stage, default 5")
    parser.add_argument("--search_type", type=str, default="ddg",  choices=["ddg", "google", 'yahoo', 'bing'],
                        help="search engine, support ddg, google, yahoo, bing")
    parser.add_argument("--max_webpage_num", type=int, default=5,
                        help="The number of webpages returned in augmentation stage, default 5")
    parser.add_argument("--max_search_nums", type=int, default=5,
                        help="The number of search results returned by search engine for each query, default 5")
    parser.add_argument("--agent_name", type=str, default="",
                        help="The agent name")
    parser.add_argument("--agent_bio", type=str, default="",
                        help="The agent bio, a short description")
    parser.add_argument("--agent_instructions", type=str, default="",
                        help="The instructions of how agent thinking, acting, or talking")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"],
                        help="The language of the overall system, default en")
    parser.add_argument("--max_tokens_num", type=int, default=4096,
                        help="Maximum number of token, default 4096")
    parser.add_argument("--num_worker", type=int, default=3,
                        help="Number of worker for multi-processing, default 3")
    parser.add_argument("--wo_tool", default=False, action='store_true',
                        help="Whether to let LLMs direct answer the query without search, default False")
    parser.add_argument("--overwrite", default=False, action='store_true',
                        help="Whether to overwrite the output path, default False")
    parser.add_argument("--lang_aware", default=False, action='store_true',
                        help="Whether to use language-aware prompt, default False")

    args = parser.parse_args()

    CFG.local_llm_host = args.local_llm_host
    CFG.local_llm_port = args.local_llm_port
    CFG.use_local_llm = args.use_local_llm

    if args.query_path:
        # process a list of queries from file
        query_data = load_queries(args.query_path)
        lang_aware_str = '_lang_aware' if args.lang_aware else ''

        # Save intermediate results
        if args.output_path is None:
            file_name = os.path.basename(args.query_path).split('.')[0]
            res_path = os.path.join(os.path.dirname(args.query_path),
                                    f'{file_name}_result_{args.lang}_{args.llm_name}_{args.max_iter_num}_{args.max_webpage_num}_wotool_{args.wo_tool}_{args.search_type}{lang_aware_str}.jsonl')
        else:
            res_path = args.output_path
        print("output path", args.output_path, res_path)

        query_key = f'query_{args.lang}'
        if os.path.exists(res_path):
            if args.overwrite:
                query_data = get_unfinished_data_and_overwrite(res_path, query_data, query_key)
            else:
                query_data = get_unfinished_data(res_path, query_data, query_key)
        else:
            print(f"No finished queries, process a new dataset with {len(query_data)} queries")

        max_retry = 3
        retry = 0
        while query_data and retry < max_retry:
            with Manager() as manager:
                lock = manager.Lock()
                with ProcessPoolExecutor(max_workers=args.num_worker) as executor:
                    future_to_query = {executor.submit(process_query, query, args, res_path, lock): query
                                       for query in query_data}
                    for future in tqdm(as_completed(future_to_query), total=len(query_data), desc="Processing"):
                        query = future_to_query[future]
                        try:
                            future.result()
                        except KeyboardInterrupt:
                            exit()
                        except Exception as e:
                            print(f"Error processing query {query['id']}: {e}")
            retry += 1
            query_data = get_unfinished_data(res_path, query_data, query_key)
        print('Results saved in', res_path)
    else:
        # process one query
        args.print_to_console = True
        agent_service = AgentService()
        agent_service.chat(vars(args))


if __name__ == "__main__":
    main()
