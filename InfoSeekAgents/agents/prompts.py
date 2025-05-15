import json

from InfoSeekAgents.utils.date_utils import get_current_time_and_date
from InfoSeekAgents.utils.function_utils import transform_to_openai_function


language_aware_planning_prompt_template = """
你是{agent_name}，{agent_bio}，{agent_instructions}
当前阶段是任务规划阶段，你被要求解答用户给定的问题，请发挥LLM的优势并且追求高效的策略进行任务规划。
1. 你有大约4000字的短期记忆
2. 用户不提供进一步的回复或者帮助
3. 规划的时候可以用参考工具中提到的工具
4. 复杂问题可以拆分成子问题后分别进行信息搜集，聚合，注意鉴别真伪消息
5. 保持谦逊，对自己没把握的问题请调用工具，但不能重复相同的调用（同样的工具和参数）
6. 你可以灵活的切换搜索词的语言以获得更多信息。可以选择使用中文、英文或者该问题涉及到的实体相关的语言搜索（比如一个法国人可以用法语搜索）
7. 你最多只能进行{max_iter_num}步思考，规划{max_iter_num}个任务，所以尽可能高效规划任务
8. 你有自我批评和反思的能力，时常反思过去的决策和策略以改进你的方法
9. 当已完成的任务已经能够得到回答给定问题的信息，则调用结束工具结束规划，否则应继续规划，但不能跟之前任务重复

{tool_specification}

{current_date_and_time}

{memory}

给定问题:{goal}

\n根据给定问题和已有任务，规划一个新任务(不能重复)，你只能以以下JSON列表的格式生成Task：
[{{
    "task_name": "任务描述",
    "command":{{
        "name":"command name",
        "args":{{
            "arg name":"value"
        }}
    }}
}}]
即使只有一个task或者没有任务也需要以列表的形式返回，确保任务（Task）可以被Python的json.loads解析

当已完成的任务已经能够帮助回答给定问题，则结束规划，否则生成一个不与之前任务重复的任务。
一个新任务:
""".strip()

language_aware_planning_prompt_template_en = """
You are a {agent_name}, {agent_bio}, {agent_instructions}
Currently, you are in the task planning phase, where you will be given a specific query to address. \
Please utilize LLM's advantages and pursue efficient strategies for task planning.\

1. You have a short-term memory of approximately 4,000 characters.
2. You do not require assistance or response from users.
3. You can use the reference tools mentioned when planning.
4. Complex problems can be split into sub-problems and then information can be collected, aggregated and authenticated. Be sure to verify the truthfulness of the information.
5. Stay humble and call the tool for questions you are not sure about, but do not call the same tool with the same parameters repeatedly.
6. You can flexibly switch the language of the search term to get more information. You can choose to search in Chinese, English, or the language related to the entity involved in the question (for example, if the question involves a French person, you can search in French)
7. You can think and plan up to {max_iter_num} steps, so strive to plan tasks as efficiently as possible.
8. You have the capability for reflection and self-criticism; reflect on past decisions and improve your strategies.
9. If you have sufficient information to answer the given query, invoke the termination tool to terminate planning. Otherwise, continue planning new tasks while ensuring no duplication with prior tasks.

{tool_specification}

{current_date_and_time}

{memory}

Given Query:{goal}

\nBased on the given question and existing tasks, plan a new Task (no repetitions), and you can only generate the Task in the following **JSON list** format:
[{{
    "task_name": "task description",
    "command":{{
        "name":"command name",
        "args":{{
            "arg name":"value"
        }}
    }}
}}]
Even if there is only one task or no task, it needs to be returned in the form of a list. Ensure that the Task can be parsed by Python's json.loads function. 

If the completed Tasks are sufficient to answer the query, terminate planning. Otherwise, create another Task that do not duplicate previous ones. 
A new Task:
""".strip()


planning_prompt_template = """
你是{agent_name}，{agent_bio}，{agent_instructions}
当前阶段是任务规划阶段，你被要求解答用户给定的问题，请发挥LLM的优势并且追求高效的策略进行任务规划。
1. 你有大约4000字的短期记忆
2. 用户不提供进一步的回复或者帮助
3. 规划的时候可以用参考工具中提到的工具
4. 复杂问题可以拆分成子问题后分别进行信息搜集，聚合，注意鉴别真伪消息
5. 保持谦逊，对自己没把握的问题请调用工具，但不能重复相同的调用（同样的工具和参数）
6. 你最多只能进行{max_iter_num}步思考，规划{max_iter_num}个任务，所以尽可能高效规划任务
7. 你有自我批评和反思的能力，时常反思过去的决策和策略以改进你的方法
8. 当已完成的任务已经能够得到回答给定问题的信息，则调用结束工具结束规划，否则应继续规划，但不能跟之前任务重复

{tool_specification}

{current_date_and_time}

{memory}

给定问题:{goal}

\n根据给定问题和已有任务，规划一个新任务(不能重复)，你只能以以下JSON列表的格式生成Task：
[{{
    "task_name": "任务描述",
    "command":{{
        "name":"command name",
        "args":{{
            "arg name":"value"
        }}
    }}
}}]
即使只有一个task或者没有任务也需要以列表的形式返回，确保任务（Task）可以被Python的json.loads解析

当已完成的任务已经能够帮助回答给定问题，则结束规划，否则生成一个不与之前任务重复的任务。
一个新任务:
""".strip()

planning_prompt_template_en = """
You are a {agent_name}, {agent_bio}, {agent_instructions}
Currently, you are in the task planning phase, where you will be given a specific query to address. \
Please utilize LLM's advantages and pursue efficient strategies for task planning.\

1. You have a short-term memory of approximately 4,000 characters.
2. You do not require assistance or response from users.
3. You can use the reference tools mentioned when planning.
4. Complex problems can be split into sub-problems and then information can be collected, aggregated and authenticated. Be sure to verify the truthfulness of the information.
5. Stay humble and call the tool for questions you are not sure about, but do not call the same tool with the same parameters repeatedly.
6. You can think and plan up to {max_iter_num} steps, so strive to plan tasks as efficiently as possible.
7. You have the capability for reflection and self-criticism; reflect on past decisions and improve your strategies.
8. If you have sufficient information to answer the given query, invoke the termination tool to terminate planning. Otherwise, continue planning new tasks while ensuring no duplication with prior tasks.

{tool_specification}

{current_date_and_time}

{memory}

Given Query:{goal}

\nBased on the given question and existing tasks, plan a new Task (no repetitions), and you can only generate the Task in the following **JSON list** format:
[{{
    "task_name": "task description",
    "command":{{
        "name":"command name",
        "args":{{
            "arg name":"value"
        }}
    }}
}}]
Even if there is only one task or no task, it needs to be returned in the form of a list. Ensure that the Task can be parsed by Python's json.loads function. 

If the completed Tasks are sufficient to answer the query, terminate planning. Otherwise, create another Task that do not duplicate previous ones. 
A new Task:
""".strip()

conclusion_prompt_template = """
你是{agent_name}，{agent_bio}，{agent_instructions}
当前阶段是总结阶段，在前几次交互中，对于用户给定的问题，你已经通过自己搜寻出了一定信息，你需要整合这些信息用中文给出最终的结论。
1. 聚焦于解决用户给定的问题
2. 如果用户给出了错误的前提，请在回答中指出

{current_date_and_time}

{memory}

给定问题：{goal}

生成该问题的简洁的中文回答:
"""

conclusion_prompt_template_en = """
You are a {agent_name}, {agent_bio}, {agent_instructions}
TThe current stage is the concluding stage. In the previous interactions, \
you have already found some information by searching on your own for the user's given query. \
You need to integrate this information and provide the final conclusion in English.
1. Focus on solving the problem given by the user
2. If the user gives a wrong premise, please point it out in your answer

{current_date_and_time}

{memory}

Given query: {goal}

Generate a brief answer **in English** for the query:
"""


ranking_prompt_template = """
你是{agent_name}，{agent_bio}，{agent_instructions}
当前阶段是网页排序阶段，在前几次交互中，对于用户给定的问题，你已经通过自己搜寻出了一些网页，你需要整合这些信息挑选出{max_webpage_num}个最相关的网页并排序。
1. 网页定义为URL和网页总结或者从网页中提取的于问题相关的重要信息
2. 搜寻的信息可能会出现冗余，避免给出相同的网页（以URL相同判定）而是聚合从相同网页得到的不同信息
3. 输出网页列表要包含回答问题必须的相关网页，如果问题有多个子问题则要包含各子问题相关网页
4. 输出网页列表中网页数量可以少于{max_webpage_num}，如果多于{max_webpage_num}则挑选出{max_webpage_num}个最重要的
5. 输出网页列表按照其对回答问题的重要程度排序，即排名第一的网页对于问题回答的贡献程度最大

{current_date_and_time}

{memory}

给定问题：{goal}

你只能以以下JSON列表的格式生成网页列表：
[{{
    "url": "网页的URL",
    "content": "从网页中提取的与给定问题相关的重要信息",
}}]
一定要返回列表，即使没有相关网页也需要返回空列表，确保任务（Task）可以被Python的json.loads解析

相关网页（按重要程度排序）:
"""

ranking_prompt_template_en = """
You are a {agent_name}, {agent_bio}, {agent_instructions}
The current stage is webpage ranking stage. In the previous interactions, \
you have already found several webpages in response to the user's query. Now, you need to consolidate this information and select the {max_webpage_num} most relevant webpages, then rank them. \
1. A webpage consists of a URL and webpage summary or information extracted from the webpage that is relevant to the query.
2. If multiple pieces of information come from the same webpage (determined by identical URLs), merge them rather than listing duplicates.
3. The output webpage list must include relevant webpages necessary to answer the question. If the question has multiple sub-questions, the relevant webpages of each sub-question must be included.
4. The number of webpages in the output webpage list can be less than {max_webpage_num}. If it is more than {max_webpage_num}, select {max_webpage_num} of the most important ones.
5. The output webpage list is sorted according to its importance to answering the question, that is, the webpage ranked first has the greatest contribution to answering the question.

{current_date_and_time}

{memory}

Given Query: {goal}

You must generate the list of webpages strictly in the following **JSON list** format:
[{{
    "url": "The webpage's URL",
    "content": "Information extracted from the webpage that is relevant to the given query",
}}]
Always return a list, even if there is no relevant web page, you need to return an empty list to ensure that the task can be parsed by Python's json.loads

Relevant webpages (ranked by importance):
"""

answer_prompt_template = """
你是{agent_name}，{agent_bio}，{agent_instructions}
当前阶段是问题回答阶段，根据你自身的知识和相关的网页信息，回答用户给定的问题。
1. 如果用户给定的问题包含多个回答，要全部列举
2. 如果用户的问题基于错误的前提，要指出错误

{current_date_and_time}

给定问题: {goal}

相关网页: {webpages}

生成解决用户问题的简洁的中文回答:
"""

answer_prompt_template_en = """
You are {agent_name}, {agent_bio}, {agent_instructions}
Currently, you are in the question-answering stage. Based on your own knowledge and relevant webpages, answer the given query from the user.
1. If the user's query contains multiple answers, list all of them
2. If the user's query is based on a wrong premise, point out the error

{current_date_and_time}

Given query: {goal}

Relevant webpages: {webpages}

Generate an brief English answer to solve the user's query:
"""


def make_planning_prompt(agent_profile, goal, used_tools, memory, max_tokens_num, tokenizer, lang="en", language_aware=False):
    tool_spec = make_tool_specification(used_tools, lang)
    if language_aware:
        template = language_aware_planning_prompt_template if lang == "zh" else language_aware_planning_prompt_template_en
    else:
        template = planning_prompt_template if lang == "zh" else planning_prompt_template_en
    prompt = template.format(**{
        "agent_name": agent_profile.name,
        "agent_bio": agent_profile.bio,
        "agent_instructions": agent_profile.instructions,
        "max_iter_num": agent_profile.max_iter_num,
        "tool_specification": tool_spec,
        "current_date_and_time": get_current_time_and_date(lang),
        "memory": memory,
        "goal": goal
    })
    prompt = prompt_truncate(tokenizer, prompt, memory, max_tokens_num)
    return prompt


def make_tool_specification(tools, lang="en"):
    functions = [transform_to_openai_function(t) for t in tools]

    commands, cnt = [], 1
    for f in functions:
        func_str = json.dumps(f, ensure_ascii=False)
        commands.append(f"{cnt}:{func_str}")
        cnt += 1

    used_commands = "\n".join(commands)

    tool_spec = f'Commands:\n{used_commands}\n'

    return tool_spec


def make_task_conclusion_prompt(agent_profile, goal, memory, max_tokens_num, tokenizer, lang="en"):
    template = conclusion_prompt_template if lang == "zh" else conclusion_prompt_template_en
    prompt = template.format(**{
        "agent_name": agent_profile.name,
        "agent_bio": agent_profile.bio,
        "agent_instructions": agent_profile.instructions,
        "current_date_and_time": get_current_time_and_date(lang),
        "memory": memory,
        "goal": goal,
    })
    prompt = prompt_truncate(tokenizer, prompt, memory, max_tokens_num)
    return prompt


def make_no_task_conclusion_prompt(query, conversation_history=""):
    prompt = ""
    if conversation_history:
        for tmp in conversation_history[-3:]:
            prompt += f"User: {tmp['query']}\nAssistant:{tmp['answer']}\n"
        prompt += f"User: {query}\nAssistant:"
    else:
        prompt = query
    return prompt


def make_task_ranking_prompt(agent_profile, goal, memory, max_tokens_num, tokenizer, max_webpage_num, lang="en"):
    template = ranking_prompt_template if lang == "zh" else ranking_prompt_template_en
    prompt = template.format(**{
        "agent_name": agent_profile.name,
        "agent_bio": agent_profile.bio,
        "agent_instructions": agent_profile.instructions,
        "current_date_and_time": get_current_time_and_date(lang),
        "memory": memory,
        "goal": goal,
        "max_webpage_num": max_webpage_num
    })
    prompt = prompt_truncate(tokenizer, prompt, memory, max_tokens_num)
    return prompt


def make_task_answer_prompt(agent_profile, goal, webpages, lang="en"):
    template = answer_prompt_template if lang == "zh" else answer_prompt_template_en
    prompt = template.format(**{
        "agent_name": agent_profile.name,
        "agent_bio": agent_profile.bio,
        "agent_instructions": agent_profile.instructions,
        "current_date_and_time": get_current_time_and_date(lang),
        "webpages": webpages,
        "goal": goal,
    })
    return prompt


def prompt_truncate(tokenizer, prompt, memory, input_max_length):
    kwargs = dict(add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, **kwargs)
    if len(prompt_tokens) > input_max_length:
        if memory is None or memory not in prompt:
            prompt_tokens = prompt_tokens[:input_max_length//2] + prompt_tokens[-input_max_length//2:]
        else:
            memory_prompt_tokens = tokenizer.encode(memory, add_special_tokens=False)
            sublst_len = len(memory_prompt_tokens)
            start_index = None
            for i in range(len(prompt_tokens) - sublst_len + 1):
                if prompt_tokens[i:i+sublst_len] == memory_prompt_tokens:
                    start_index = i
                    break
            
            if start_index is None:
                prompt_tokens = prompt_tokens[:input_max_length//2] + prompt_tokens[-input_max_length//2:]
            else:
                other_len = len(prompt_tokens) - sublst_len
                if input_max_length > other_len:
                    max_memory_len = input_max_length - other_len
                    memory_prompt_tokens = memory_prompt_tokens[:max_memory_len//2] + memory_prompt_tokens[-max_memory_len//2:]
                    prompt_tokens = prompt_tokens[:start_index] + memory_prompt_tokens + prompt_tokens[start_index + sublst_len:]
    prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    return prompt