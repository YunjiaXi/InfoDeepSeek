export API_KEY="your_api_key_for_deepseek"
export API_TYPE=deepseek
export API_BASE=https://api.deepseek.com
#export http_proxy='http://127.0.0.1:7890'  # sometimes need proxy


lang="en"
query='When did the country ranked 221st in the 2017 list of countries and regions by population gain independence?'
llm='deepseek-chat'
fast_llm='deepseek-chat'
max_iter_num=5

python -m InfoSeekAgents.agent_start --query "${query}" --lang ${lang} --llm_name ${llm} \
      --max_iter_num ${max_iter_num} --fast_llm_name ${fast_llm}

