from collections import deque
import json
import logging
import re
import os
import sys
import time
import traceback
from typing import Dict, List
import uuid
from transformers import AutoTokenizer
from pathlib import Path

from InfoSeekAgents.tools import ALL_NO_TOOLS, ALL_TOOLS, FinishTool, NoTool
from InfoSeekAgents.llms import create_chat_completion
from InfoSeekAgents.agents.prompts import make_planning_prompt, make_task_answer_prompt, make_task_ranking_prompt
from InfoSeekAgents.agents.prompts import make_no_task_conclusion_prompt, make_task_conclusion_prompt
from InfoSeekAgents.utils.chain_logger import *
from InfoSeekAgents.utils.json_fix_general import find_json_dict, correct_json, find_json_list


class SingleTaskListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]
    
    def get_tasks(self):
        return list(self.tasks)

    def clear(self):
        del self.tasks
        self.tasks = deque([])
        self.task_id_counter = 0


class InfoSeekAgent(object):
    def __init__(self, cfg, session_id=None, agent_profile=None, tools=None, lang="en"):
        self.cfg = cfg
        self.agent_profile = agent_profile
        self.lang = lang
        self.max_task_num = agent_profile.max_iter_num
        self.session_id = session_id if session_id else str(uuid.uuid1())
        self.tokenizer = self.initialize_tokenizer(self.cfg.fast_llm_model)

        self.initialize_logger()
        self.initialize_memory()
        self.tool_retrival(tools)

    def initialize_logger(self):
        self.chain_logger = ChainMessageLogger(output_streams=[sys.stdout], lang=self.lang,
                                               print_to_console=self.cfg.print_to_console)
        self.cfg.set_chain_logger(self.chain_logger)

    def initialize_memory(self):
        pass
    
    def initialize_tokenizer(self, llm_name):
        if "baichuan" in llm_name:
            model_name = "kwaikeg/kagentlms_baichuan2_13b_mat"
        elif "qwen_7b" in llm_name:
            model_name = "kwaikeg/kagentlms_qwen_7b_mat"
        else:
            model_name = str(Path(__file__).parent.parent / 'resources/gpt2')
        print(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            padding_side='left',
            trust_remote_code=True
        )
        return tokenizer

    def tool_retrival(self, tools):
        if tools:
            self.tools = [tool_cls(cfg=self.cfg) for tool_cls in tools]
        else:
            if "notool" in self.agent_profile.tools:
                self.tools = list()
            else:
                all_tools = [tool_cls(cfg=self.cfg) for tool_cls in ALL_TOOLS]

                if "auto" in self.agent_profile.tools:
                    # used_tools = [tool_cls(cfg=self.cfg) for tool_cls in ALL_TOOLS]
                    used_tools = all_tools
                else:
                    used_tools = list()
                    for tool in all_tools:
                        if tool.zh_name in self.agent_profile.tools or tool.name in self.agent_profile.tools:
                            used_tools.append(tool)
                used_tools += [tool_cls(cfg=self.cfg) for tool_cls in ALL_NO_TOOLS]
            
            self.tools = used_tools
        self.name2tools = {t.name: t for t in self.tools}

    def memory_retrival(self, 
        goal: str, 
        conversation_history: List[List], 
        complete_task_list: List[Dict]):

        memory = ""
        if conversation_history:
            memory += f"* Conversation History:\n"
            for tmp in conversation_history[-3:]:
                memory += f"User: {tmp['query']}\nAssistant:{tmp['answer']}\n"

        if complete_task_list:
            complete_task_str = json.dumps(complete_task_list, ensure_ascii=False, indent=4)
            memory += f"* Complete tasks: {complete_task_str}\n"
        return memory

    def task_plan(self, goal, memory):
        prompt = make_planning_prompt(self.agent_profile, goal, self.tools, memory, self.cfg.max_tokens_num,
                                      self.tokenizer, lang=self.lang, language_aware=self.cfg.lang_aware)
        try:
            response, _ = create_chat_completion(
                query=prompt, llm_model_name=self.cfg.smart_llm_model)
            self.chain_logger.put_prompt_response(
                prompt=prompt, 
                response=response, 
                session_id=self.session_id, 
                mtype="auto_task_create",
                llm_name=self.cfg.smart_llm_model)
            response = correct_json(find_json_list(response))
            task = json.loads(response)
            new_tasks = task
        except KeyboardInterrupt:
            exit()
        except:
            print(traceback.format_exc())
            self.chain_logger.put("fail", logging_think_fail_msg(self.lang))
            new_tasks = {}
        return new_tasks

    def tool_use(self, command) -> str:
        try:
            command_name = command.get("name", "")
            if command_name == "search":
                command_name = "web_search"
            args_text = ",".join([f'{key}={val}' for key, val in command["args"].items()])
            execute_str = f'{command_name}({args_text})'.replace("wikipedia(", "kuaipedia(")
            self.chain_logger.put("execute", execute_str)
            if not command_name:
                raise RuntimeError("{} has no tool name".format(command))
            if command_name not in self.name2tools:
                raise RuntimeError("has no tool named {}".format(command_name))
            tool = self.name2tools[command_name]

            tool_output = tool(**command["args"])
            self.chain_logger.put("observation", tool_output.answer_md)

            for prompt, response in tool_output.prompt_responses:
                self.chain_logger.put_prompt_response(
                    prompt=prompt,
                    response=response,
                    session_id=self.session_id,
                    mtype=f"auto_command_{command_name}",
                    llm_name=self.cfg.fast_llm_model
                )
            return tool_output.answer
        except KeyboardInterrupt:
            exit()
        except:
            print(traceback.format_exc())
            self.chain_logger.put("observation", logging_execute_fail_msg(self.lang))
            return ""

    def conclusion(self, 
                   goal: str,
                   memory,
                   conversation_history: List[List],
                   is_rank: bool=True,
                   max_webpage_num: int = 5,
                   no_task_planned: bool = False
                   ):

        if no_task_planned:
            prompt = make_no_task_conclusion_prompt(goal, conversation_history)
        else:
            if is_rank:
                prompt = make_task_ranking_prompt(self.agent_profile, goal, memory, self.cfg.max_tokens_num, self.tokenizer, max_webpage_num, lang=self.lang)
            else:
                prompt = make_task_conclusion_prompt(self.agent_profile, goal, memory, self.cfg.max_tokens_num, self.tokenizer, lang=self.lang)

        response, _ = create_chat_completion(
            query=prompt, 
            chat_id="kwaiagents_conclude_" + self.session_id,
            llm_model_name=self.cfg.smart_llm_model)

        self.chain_logger.put_prompt_response(
            prompt=prompt, 
            response=response, 
            session_id=self.session_id, 
            mtype="auto_conclusion",
            llm_name=self.cfg.smart_llm_model)
        return response

    def answer(self, goal, webpages, k):
        prompt = make_task_answer_prompt(self.agent_profile, goal, webpages[:k], lang=self.lang)
        # print(f'\n************** ANSWER AGENT PROMPT {k}*************')
        # print(prompt)

        response, _ = create_chat_completion(
            query=prompt,
            chat_id=f"kwaiagents_answer_{k}_{self.session_id}",
            llm_model_name=self.cfg.smart_llm_model)

        self.chain_logger.put_prompt_response(
            prompt=prompt,
            response=response,
            session_id=self.session_id,
            mtype="auto_answer",
            llm_name=self.cfg.smart_llm_model)
        return response

    def check_task_complete(self, task, iter_id):
        command_name = task["command"]["name"]
        if not task or ("task_name" not in task) or ("command" not in task) \
            or ("args" not in task["command"]) or ("name" not in task["command"]):
            self.chain_logger.put("finish", str(task.get("task_name", "")))
            return True
        elif command_name == FinishTool.name:
            self.chain_logger.put("finish", str(task["command"]["args"].get("reason", "")))
            return True
        elif command_name == NoTool.name:
            if iter_id == 1:
                self.chain_logger.put("finish", logging_do_not_need_use_tool_msg(self.lang))
            else:
                self.chain_logger.put("finish", logging_do_not_need_use_tool_anymore_msg(self.lang))
            return True
        elif command_name not in self.name2tools:
            self.chain_logger.put("finish", logging_do_not_need_use_tool_msg(self.lang))
            return True
        else:
            return False

    def chat(self, query, history=list(), max_webpage_num=5, *args, **kwargs):
        goal = query
        res_info = {}
        new_history = []
        offline_conclusion = None
        if not self.tools:
            no_task_planned = True
            conclusion = self.conclusion(
                goal,
                memory="",
                conversation_history=history,
                no_task_planned=no_task_planned)
            self.chain_logger.put("chain_end", "")

            new_history = history[:] + [{"query": query, "answer": conclusion}]
        else:
            tasks_storage = SingleTaskListStorage()
            tasks_storage.clear()

            start = True
            loop = True
            iter_id = 0
            complete_task_list = list()
            no_task_planned = False
            while loop:
                iter_id += 1
                if start or not tasks_storage.is_empty():
                    start = False
                    if not tasks_storage.is_empty():
                        task = tasks_storage.popleft()
                        
                        if (self.check_task_complete(task, iter_id,)):
                            if iter_id <= 2:
                                no_task_planned = True
                            break

                        self.chain_logger.put("thought", task.get("task_name", ""))

                        result = self.tool_use(task["command"])

                        task["result"] = result
                        complete_task_list.append(task)

                    if iter_id > self.agent_profile.max_iter_num:
                        self.chain_logger.put("finish", logging_stop_thinking_msg(self.lang))
                        break
                    self.chain_logger.put("thinking")
                    memory = self.memory_retrival(goal, history, complete_task_list)
                    new_tasks = self.task_plan(goal, memory)

                    for new_task in new_tasks:
                        new_task.update({"task_id": tasks_storage.next_task_id()})
                        tasks_storage.append(new_task)
                    # print(complete_task_list)
                else:
                    loop = False
                    self.chain_logger.put("finish", logging_finish_task_msg(self.lang))

            memory = self.memory_retrival(goal, history, complete_task_list)

            conclusion = self.conclusion(
                goal,
                memory=memory,
                conversation_history=history,
                is_rank=False,
                max_webpage_num=max_webpage_num,
                no_task_planned=no_task_planned)

            self.chain_logger.put("conclusion", "online:\n" + conclusion)

            if self.cfg.wo_tool:
                offline_conclusion = self.conclusion(
                    goal,
                    memory="",
                    conversation_history=history,
                    no_task_planned=True)

                self.chain_logger.put("conclusion", "offline:\n" + offline_conclusion)

            webpage_resp = self.conclusion(
                goal,
                memory=memory,
                conversation_history=history,
                is_rank=True,
                max_webpage_num=max_webpage_num,
                no_task_planned=no_task_planned)

            webpages = json.loads(correct_json(find_json_list(webpage_resp)))

            self.chain_logger.put("ranking", json.dumps(webpages, ensure_ascii=False))
            conclusion_k = {}
            for k in range(len(webpages)):
                answer = self.answer(goal, webpages=webpages, k=k+1)
                conclusion_k[k+1] = answer
                self.chain_logger.put(f"answer", f"answer_at_{k+1}:\n" + answer)

            if len(webpages) == 0:
                for k in range(max_webpage_num):
                    conclusion_k[k + 1] = None

            self.chain_logger.put("chain_end", "")

            new_history = history[:] + [{"query": query, "answer": conclusion}]
            res_info = {
                "ranked_webpages": webpages,
                "answer_at_k": conclusion_k,
                "offline_response": offline_conclusion,  # directly answer without online search
            }

        return {
            "response": conclusion,  # answer with online search
            "history": new_history,
            "chain_msg": self.chain_logger.chain_msgs,
            "chain_msg_str": self.chain_logger.chain_msgs_str,
            "full_llm_prompt_responses": self.chain_logger.llm_prompt_responses,
            "more_info": res_info,
        }
