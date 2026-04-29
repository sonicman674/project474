export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 256 --mode train --dataset PSM --data_path dataset/PSM --input_c 25 --output_c 25 --training_log_path training_logs
python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 256 --mode test --dataset PSM --data_path dataset/PSM --input_c 25 --output_c 25 --score_save_path test_outputs


