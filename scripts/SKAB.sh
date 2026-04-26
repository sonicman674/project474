export CUDA_VISIBLE_DEVICES=0

python3 main.py \
  --anormly_ratio 1.0 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode train \
  --dataset SKAB \
  --data_path dataset/SKAB \
  --input_c 8 \
  --output_c 8

python3 main.py \
  --anormly_ratio 1.0 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset SKAB \
  --data_path dataset/SKAB \
  --input_c 8 \
  --output_c 8
