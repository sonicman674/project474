export CUDA_VISIBLE_DEVICES=0

python3 main.py \
  --anormly_ratio 1.0 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode train \
  --dataset MITBIH \
  --data_path dataset/MITBIH \
  --input_c 2 \
  --output_c 2

python3 main.py \
  --anormly_ratio 7.0 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset MITBIH \
  --data_path dataset/MITBIH \
  --input_c 2 \
  --output_c 2
