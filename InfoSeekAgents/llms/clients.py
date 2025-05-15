import os
import requests
import traceback
import openai
import google.generativeai as genai
from google.generativeai.types import RequestOptions
from google.api_core import retry


def get_qwen_response(client, model, msgs, temperature):
    reasoning_content = ""
    content = ""

    completion = client.chat.completions.create(
        model=model,
        messages=msgs,
        stream=True,
        temperature=temperature,
        # extra_body={"enable_thinking": False}, #qwen3-32b默认开思考模式，qwen-plus-latest默认不开
        # Uncomment the following line to return token usage in the last chunk
        # stream_options={
        #     "include_usage": True
        # }
    )
    for chunk in completion:
        # If chunk.choices is empty, print usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # omit reasoning content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
            else:
                content += delta.content
    return content


def make_gpt_messages(query, system, history):
    msgs = list()
    if system:
        msgs.append({
            "role": "system",
            "content": system
        })
    for q, a in history:
        msgs.append({
            "role": "user",
            "content": str(q)
        })
        msgs.append({
            "role": "assistant",
            "content": str(a)
        })
    msgs.append({
        "role": "user",
        "content": query
    })
    return msgs


class RemoteClient(object):
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.api_type = os.environ.get("API_TYPE", "open_ai")
        self.api_key = os.environ["API_KEY"]

    def chat(self, query, history=list(), system="", temperature=0.0, enable_thinking=False, stop="", *args, **kwargs):
        if self.api_type == 'google':
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(query,
                                                  request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)))
                response_text = response.text
            except:
                # print('current query', query)
                print(traceback.format_exc())
                if 'rate-limits' in traceback.format_exc():
                    exit()
                response_text = ""
                if "content_filter" in traceback.format_exc():
                    response_text = '[]'
        else:
            msgs = make_gpt_messages(query, system, history)
            try:
                if self.api_type == "azure":
                    client = openai.AzureOpenAI(
                        api_key=self.api_key,
                        api_version=os.environ.get("API_VERSION"),
                        azure_endpoint=os.environ.get("API_BASE")
                    )
                elif self.api_type == "open_ai":
                    client = openai.OpenAI(api_key=self.api_key)
                else:  # deepseek
                    client = openai.OpenAI(api_key=self.api_key, base_url=os.environ.get("API_BASE"))

                if self.api_type == "qwen":
                    response_text = get_qwen_response(client, self.model, msgs, temperature)
                else:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=msgs,
                        temperature=temperature,
                        stream=False
                    )
                    response_text = response.choices[0].message.content
            except:
                # print('current query', query)
                err = traceback.format_exc()
                print(err)
                response_text = ""
                if "content_filter" in err or "inappropriate content" in err or 'Content Exists Risk' in err:
                    response_text = '[]'

        new_history = history[:] + [[query, response_text]]
        return response_text, new_history


class FastChatClient(object):
    def __init__(self, model="llama3.3-8b", host="localhost", port=8888):
        self.model = model
        self.host = host
        self.port = port

    def chat(self, query, history=list(), system="", temperature=0.0, stop="", *args, **kwargs):
        url = f'http://{self.host}:{self.port}/v1/completions/'

        headers = {"Content-Type": "application/json"}
        if "baichuan" in self.model:
            prompt = self.make_baichuan_prompt(query, system, history)
        elif "qwen" in self.model:
            prompt = self.make_qwen_prompt(query, system, history)
        else:
            prompt = self.make_prompt(query, system, history)
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.75,
            "top_k": 40,
            "max_tokens": 512
        }
        resp = requests.post(url=url, json=data, headers=headers)
        response = resp.json() # Check the JSON Response Content documentation below
        response_text = response['choices'][0]['text']

        new_history = history[:] + [[query, response_text]]
        return response_text, new_history

    @staticmethod
    def make_prompt(query, system, history):
        if not history:
            history = list()
        if system:
            prompt = system + "\n"
        else:
            prompt = ''
        for q, r in history:
            prompt += 'User:' + q + '\nAssistant' + r + "\n"
        prompt += query
        return prompt

    @staticmethod
    def make_baichuan_prompt(query, system, history):
        if not history:
            history = list()
        if system:
            prompt = system + "\n"
        else:
            prompt = ''
        for q, r in history:
            prompt += '<reserved_106>' + q + '<reserved_107>' + r 
        prompt += query
        return prompt

    @staticmethod
    def make_qwen_prompt(query, system, history):
        if not history:
            history = list()
        if system:
            prompt = '<|im_start|>' + system + '<|im_end|>\n'
        else:
            prompt = ''
        for q, r in history:
            response = r if r else ''
            prompt += '<|im_start|>user\n' + q + '<|im_end|>\n<|im_start|>assistant\n' + response + "<|im_end|>\n"
        prompt += query
        return prompt
