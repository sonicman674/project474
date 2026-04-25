export CUDA_VISIBLE_DEVICES=0

python3 main.py \
  --anormly_ratio 1.0 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode train \
  --dataset GECCO \
  --data_path dataset/GECCO \
  --input_c 9 \
  --output_c 9

python3 main.py \
  --anormly_ratio 0.002 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset GECCO \
  --data_path dataset/GECCO \
  --input_c 9 \
  --output_c 9
