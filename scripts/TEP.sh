export CUDA_VISIBLE_DEVICES=0

python3 main.py \
  --anormly_ratio 1.0 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode train \
  --dataset TEP \
  --data_path dataset/TEP \
  --input_c 52 \
  --output_c 52

python3 main.py \
  --anormly_ratio 1.0 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset TEP \
  --data_path dataset/TEP \
  --input_c 52 \
  --output_c 52
