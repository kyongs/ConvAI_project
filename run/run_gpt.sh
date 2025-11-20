eval_path='./data/dev.json'
db_root_path='./data/dev_databases/'
mode='dev' 
engine='gpt-4o-mini'
data_output_path='./exp_result/result/'

YOUR_API_KEY=''




echo 'generate GPT4o-mini batch with knowledge'
python3 -u ./src/gpt_request.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
--engine ${engine} --eval_path ${eval_path} --data_output_path ${data_output_path} --use_knowledge True \
--chain_of_thought False

