<p align="left">
    <a href="README.md">English</a> ｜ 中文 
</p>
<br><br>

# InfoDeepSeek Benchmark

InfoDeepSeek是一个专为真实网络环境中的Agent的信息搜寻任务量身打造的基准测试，提供245条人工收集和检验的困难问题和动态环境下信息搜寻质量的评估指标。 基准测试中的问题满足以下要求
  * 确定性：有确定且唯一的答案，不随时间变化，便于在动态环境下评估。
  * 困难性：最先进的模型也难以在单轮搜索下回答成功，需要agent多轮规划和信息搜寻能力。
  * 多样性：覆盖不同的问题类型（多跳、长尾、时间敏感、新鲜事、干扰信息、虚假前提）、领域和优势语言，以覆盖真实环境中各种问题。

更多的细节见我们的论文：[InfoDeepSeek: Benchmarking Agentic Information
Seeking for Retrieval-Augmented Generation](https://arxiv.org/pdf/2505.15872)。


## 动态
* 2025.5.23 - [论文](https://arxiv.org/pdf/2505.15872)发布
* 2025.5.15 - 项目公开

## 数据集
你可以到[huggingface](https://huggingface.co/datasets/yoga334/InfoDeekSeek)下载和查看我们数据集的更多细节，也可以在[这里](data/InfoDeepSeek_v1.json)找到我们数据集。

## 使用指南

### 环境安装
下载本代码仓库并安装环境，推荐使用miniconda和Python>=3.10
 
```bash
conda create -n InfoSeekAgent python=3.10
conda activate InfoSeekAgent
pip install -r requirements.txt
```

browse_website和search工具基于Selenium实现，需要安装[Chrome](https://www.google.com/intl/en_uk/chrome/?brand=FKPE&ds_kid=43700081222624393&gad_source=1&gad_campaignid=22008060471&gbraid=0AAAAAoY3CA52gUGf-H4dvuNwzmdq8WrFA&gclid=CjwKCAjw_pDBBhBMEiwAmY02Nk6jhBuzZYteaKOJpBucORZe9ef2NhL8fxNXhfwvmmE1nchOmp_E8RoCcFoQAvD_BwE&gclsrc=aw.ds)和配置[Chromedriver](https://developer.chrome.com/docs/chromedriver/get-started)
### 基于API调用
目前我们的代码支持OpenAI、Gemini、DeepSeek和Qwen的官方API，可以[这里](InfoSeekAgents/llms/clients.py)修改代码增加API。

1. **单个问题**
```bash
sh script/run_toy.sh
```
run_toy.sh中的样例基于DeepSeek实现，具体代码如下。需要填写DeepSeek官方的API KEY。我们默认的搜索引擎是DuckDuckGo，它基于API的调用是免费的，但是可能会有速率限制。如果多次调用失败可能是网络无法访问DuckDuckGo，可以通过设置http_proxy解决。
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

2. **处理和评测来自文件的多个query**

除了单个问题直接调用，代码还支持传入json文件进行整体评测，格式参考我们的[数据](data/InfoDeepSeek_v1.json)。
我们在script中给出了一些例子， 比如对于Gemini-2.5-Flash：
```bash
sh script/run_gemini.sh
```
run_gemini.sh的内容包含了Agent信息搜寻和评估两个过程，前一个过程需要Gemini的API KEY，后一个需要DeepSeek的API KEY评估答案准确性（你也可以使用别的模型）。file_dir默认是我们benchmark的[数据](data/InfoDeepSeek_v1.json)，你可以更换其他的数据或者使用我们的toy数据集[data/toy.json](data/toy.json)，它只包含4个样例，可以用于测试环境是否正常。
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

对于DeepSeek-V3：
```bash
sh script/run_deepseek.sh
```
对于Qwen3-32B：
```bash
sh script/run_qwen.sh
```
对于GPT-4o：
```bash
sh script/run_gpt.sh
```

3. **切换搜索引擎**

我们默认的搜索引擎是DuckDuckGo，也支持其他的主流搜索引擎包括Google、Yahoo和Bing。但是他们都不是免费的，而是需要付费的API。因此，我们同时支持了这些搜索引擎使用API或者Selenium获取搜索结果。 在具体实现中，我们会首先尝试使用API获取搜索结果，如果没有API KEY或者无法通过API爬取，再基于Selenium爬取搜索结果。

使用Google搜索：
```bash
sh script/run_google.sh
```
我们使用[Serper](https://serper.dev/)的API来获取Google搜索的结果，你需要注册并获取API KEY，然后将SERPER_API_KEY填写为你自己的API KEY。

你也可以将SERPER_API_KEY留空，这样就就会跳过调用API获取搜索结果，而是用Selenium获取搜索结果（这样有爬取不成功的风险）。

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

Yahoo和Bing则基于[SerpApi](https://serpapi.com/)实现，同样需要注册和获取API KEY。对于Bing：
```bash
sh script/run_bing.sh
```
对于Yahoo：
```bash
sh script/run_yahoo.sh
```

## 引用
```
@article{xi2025infodeepseek,
        title={InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation},
        author={Yunjia Xi and Jianghao Lin and Menghui Zhu and Yongzhao Xiao and Zhuoying Ou and Jiaqi Liu and Tong Wan and Bo Chen and Weiwen Liu and Yasheng Wang and Ruiming Tang and Weinan Zhang and Yong Yu},
        year={2025},
        journal={arXiv preprint arXiv:2505.15872},
        url={https://arxiv.org/abs/2505.15872},
}
```

## License
本项目采用[CC BY-NC 4.0](LICENSE)进行许可。该许可协议允许在非商业目的下对作品进行共享和改编，但须给予适当的署名。如需快速了解，请访问[知识共享许可协议](https://creativecommons.org/licenses/by-nc/4.0/)。
