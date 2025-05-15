<p align="left">
    English ｜ <a href="README_ZH.md">中文</a> 
</p>
<br><br>

# InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation

InfoDeepSeek is a benchmark with challenging questions and novel evaluation metrics tailored for agentic information seeking tasks under real-world web environments.

The questions in the benchmark meet the following criteria:
* Determinism: The question has a definite and unique answer, unaffected by time, making it easier to evaluate in dynamic environments.
* Difficulty: Even the most advanced models struggle to answer successfully with a single round of search, requiring multi-round planning and information retrieval abilities.
* Diversity: It covers a wide range of question types (multi-hop, long-tail, time-sensitive, fresh events, distractions, false premises), domains, and major languages, addressing various real-world issues.

## News

* 2025.5.15 - Public release

## Dataset
You can download and view more details about our dataset on [huggingface](https://huggingface.co/datasets/yoga334/InfoDeekSeek), or find it [here](data/InfoDeepSeek_v1.json). 


## User Guide

### Environment setup

Download this code repository and set up the environment with miniconda. Python>=3.10 is recommended.
 
```bash
conda create -n InfoSeekAgent python=3.10
conda activate InfoSeekAgent
pip install -r requirements.txt
```

The browse_website and search tools are implemented with Selenium and require installing [Chrome](https://www.google.com/intl/en_uk/chrome/?brand=FKPE&ds_kid=43700081222624393&gad_source=1&gad_campaignid=22008060471&gbraid=0AAAAAoY3CA52gUGf-H4dvuNwzmdq8WrFA&gclid=CjwKCAjw_pDBBhBMEiwAmY02Nk6jhBuzZYteaKOJpBucORZe9ef2NhL8fxNXhfwvmmE1nchOmp_E8RoCcFoQAvD_BwE&gclsrc=aw.ds) and [Chromedriver](https://developer.chrome.com/docs/chromedriver/get-started).


### How to run and evaluate

Our code currently supports the official APIs for OpenAI, Gemini, DeepSeek, and Qwen. You can modify or add clients [Here](InfoSeekAgents/llms/clients.py).

1. Process **Single Query from CLI**
```bash
sh script/run_toy.sh
```
The run_toy.sh example uses DeepSeek; you need to set your DeepSeek API key. By default, DuckDuckGo is used as the search engine (free API but may have rate limits). If DuckDuckG fail repeatedly due to network issues, you can set http_proxy.

```bash
export API_KEY="your_api_key_for_deepseek"
export API_TYPE=deepseek
export API_BASE=https://api.deepseek.com
# export http_proxy='http://127.0.0.1:7890'  # sometimes need proxy, change 7890 to your port.

lang="en"
query='When did the country ranked 221st in the 2017 list of countries and regions by population gain independence?'
llm='deepseek-chat'
fast_llm='deepseek-chat'
max_iter_num=5

python -m InfoSeekAgents.agent_start --query "${query}" --lang ${lang} --llm_name ${llm} \
      --max_iter_num ${max_iter_num} --fast_llm_name ${fast_llm}
```

2. Process and evaluate **Multiple Queries from File**

 In addition to single‑query mode, you can evaluate an entire JSON file (format as in our [data](data/InfoDeepSeek_v1.json)). Scripts under `script/` provide examples—for instance, for Gemini‑2.5‑Flash:
 ```bash
sh script/run_gemini.sh
```

The script `run_gemini.sh` involves both the agentic information seeking and the evaluation steps. The information seeking step requires your Gemini API key, and the evaluation step uses your DeepSeek API key to assess answer accuracy (you may substitute another model). The default `file_dir` points to our benchmark data; you can replace it with your own. Or you can use our toy dataset [data/toy.json](data/toy.json), which contains only 4 examples and can be used to verify that your environment is working correctly.

```bash
export API_KEY='your_api_key_for_gemini'
export API_TYPE=google
export API_BASE=https://api.deerapi.com/v1/chat/completions
#export http_proxy='http://127.0.0.1:7890'  # sometimes need proxy

lang='zh'  # or 'en'
llm='gemini-2.5-flash-preview-04-17'  # or 'gemini-2.5-pro-exp-03-25'
fast_llm='gemini-2.0-flash'
file_dir='data/InfoDeepSeek_v1'  # do not add .json
max_iter_num=5


python -m InfoSeekAgents.agent_start --query_path ${file_dir}.json \
      --lang ${lang} --llm_name ${llm} --fast_llm_name ${fast_llm} --max_iter_num ${max_iter_num} \
      --wo_tool --num_worker=1 --overwrite

export API_KEY="your_api_key_for_deepseek"
export API_TYPE=deepseek
export API_BASE=https://api.deepseek.com
eval_llm='deepseek-chat'

python eval/eval.py --input_file ${file_dir}_result_${lang}_${llm}_${max_iter_num}_5_wotool_True_ddg.jsonl \
      --eval_llm_name ${eval_llm} --reuse

python eval/cal_acc.py --input_file ${file_dir}_result_${lang}_${llm}_${max_iter_num}_5_wotool_True_ddg_${eval_llm}_score.jsonl \
      --print_attribute

```

For DeepSeek-V3：
```bash
sh script/run_deepseek.sh
```
For Qwen3-32B：
```bash
sh script/run_qwen.sh
```
For GPT-4o：
```bash
sh script/run_gpt.sh
```

3. **Switching Search Engines**

Our default search engine is DuckDuckGo, but you can also use Google, Yahoo, or Bing, though these require paid APIs. We support both API-based and Selenium-based search. By default, we try API first; if no key is available or API calls fail, we fall back to Selenium.

For Google Search:
```bash
sh script/run_google.sh
```
We use [Serper](https://serper.dev/)'s API for Google; set SERPER_API_KEY to your key. If left empty, Selenium scraping will be used (with possible failures).


```bash
export API_KEY="your_api_key_for deepseek"
export API_TYPE=deepseek
export API_BASE=https://api.deepseek.com
export http_proxy='http://127.0.0.1:7890'  # sometimes need proxy

export SERPER_API_KEY="your_api_key_for_serper"  # for google search

lang="zh" # or "en"
llm='deepseek-chat'  # 'deepseek-reasoner'
fast_llm='deepseek-chat'
eval_llm='deepseek-chat'
max_iter_num=5
file_dir='./data/InfoDeepSeek_v1'  # do not add .json
search_type='google'


python -m InfoSeekAgents.agent_start --query_path ${file_dir}.json  --fast_llm_name ${fast_llm} \
      --lang ${lang} --llm_name ${llm} --max_iter_num ${max_iter_num} --wo_tool --num_worker=1 --search_type=${search_type} --overwrite

python eval/eval.py --input_file ${file_dir}_result_${lang}_${llm}_${max_iter_num}_5_wotool_True_${search_type}.jsonl \
      --eval_llm_name ${eval_llm} --reuse --num_worker=2

python eval/cal_acc.py --input_file ${file_dir}_result_${lang}_${llm}_${max_iter_num}_5_wotool_True_${search_type}_${eval_llm}_score.jsonl \
      --print_attribute
```

Yahoo and Bing use [SerpApi](https://serpapi.com/) and also require API keys. For Bing:
```bash
sh script/run_bing.sh
```

For Yahoo：
```bash
sh script/run_yahoo.sh
```

[//]: # (## Citations)


## License
This project is licensed u[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work for non-commercial purposes, provided appropriate attribution is given. For a quick overview, please visit the [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).