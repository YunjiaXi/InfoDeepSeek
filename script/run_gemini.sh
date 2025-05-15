export API_KEY="your_api_key_for_gemini"
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
