export API_KEY="your_api_key_for_deepseek"
export API_TYPE=deepseek
export API_BASE=https://api.deepseek.com
#export http_proxy='http://127.0.0.1:7890'  # sometimes need proxy

export SERP_API_KEY="your_api_key_for_serpapi"

lang="zh" # or "en"
llm='deepseek-chat'  # 'deepseek-reasoner'
fast_llm='deepseek-chat'
eval_llm='deepseek-chat'
max_iter_num=5
file_dir='data/InfoDeepSeek_v1'  # do not add .json
search_type='yahoo'


python -m InfoSeekAgents.agent_start --query_path ${file_dir}.json  --fast_llm_name ${fast_llm} \
      --lang ${lang} --llm_name ${llm} --max_iter_num ${max_iter_num} --wo_tool --num_worker=1 --search_type=${search_type} --overwrite

python eval/eval.py --input_file ${file_dir}_result_${lang}_${llm}_${max_iter_num}_5_wotool_True_${search_type}.jsonl \
      --eval_llm_name ${eval_llm} --reuse --num_worker=2

python eval/cal_acc.py --input_file ${file_dir}_result_${lang}_${llm}_${max_iter_num}_5_wotool_True_${search_type}_${eval_llm}_score.jsonl \
      --print_attribute

