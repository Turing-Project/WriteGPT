#!/usr/bin/env bash

export PYTHONPATH=../

learning_rate=1e-4
init_checkpoint="gs://models/model.ckpt-220000"
max_seq_length=1024
save_checkpoint_steps=1000

# You can customize the training here
# mega, medium, or base
model_type="mega"
OUTPUT_DIR="gs://trained_models" # put your output directory here
input_file="gs://data_input/Essay_Killer_0000.tfrecord" # put your input files here, it can also be something like "*.tfrecord"

if [ ${model_type} == "base" ]; then
    num_tpu_cores=32
    batch_size_per_core=16
elif [ ${model_type} == "medium" ]; then
    num_tpu_cores=128
    batch_size_per_core=4
elif [ ${model_type} == "mega" ]; then
    num_tpu_cores=256
    batch_size_per_core=2


# there are 20k * 1024 examples so this translates to 20 epochs. seems ok and i can run for more if needed
num_train_steps=800000

# Make sure batch size scales.
let batch_size="$batch_size_per_core * $num_tpu_cores"

python train_tpu.py \
    --config_file=configs/${model_type}.json \
    --input_file=${input_file} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${max_seq_length} \
    --train_batch_size=${batch_size} \
    --learning_rate=${learning_rate} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=10000 \
    --save_checkpoints_steps=${save_checkpoint_steps} \
    --iterations_per_loop=${save_checkpoint_steps} \
    --use_tpu=True \
    --tpu_name=$(hostname) \
    --num_tpu_cores=$num_tpu_cores \
    --init_checkpoint=${init_checkpoint}
