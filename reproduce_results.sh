# Get activations
cd get_activations
python get_activations.py --model_name llama3_8B_instruct --dataset_name tqa_mc2
python get_activations.py --model_name llama3_8B_instruct --dataset_name tqa_gen
python get_activations.py --model_name llama3_8B_instruct --dataset_name tqa_gen_end_q

# Validation
cd ../validation
python validate_2fold.py --model_name llallama3_8B_instructm --beta 0.0 --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default
python validate_2fold.py --model_name llallama3_8B_instructm --beta 1.0 --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default
python validate_2fold.py --model_name llallama3_8B_instructm --beta 2.0 --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default
python validate_2fold.py --model_name llallama3_8B_instructm --beta 5.0 --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default
python validate_2fold.py --model_name llallama3_8B_instructm --beta 10.0 --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default

python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_0_beta_0.0_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_0_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_0_beta_2.0_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_0_beta_5.0_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_0_beta_10.0_com.csv

python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_1_beta_0.0_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_1_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_1_beta_2.0_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_1_beta_5.0_com.csv
python evaluate_with_llama_judges.py --input_path results_dump/answer_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15_fold_1_beta_10.0_com.csv

# Next, analyse results with results_analysis.ipynb

