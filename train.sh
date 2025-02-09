export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/x86_64-linux-gnu/
export PATH=$PATH:$HOME/.local/bin
python train_pearl_generator.py \
    --model_name_or_path models/flan-t5-large \
    --output_dir models/PEARL_generator_for_MPT-7B_on_QA \
    --evaluation_strategy no \
    --save_strategy no \
    --do_train \
    --do_eval \
    --train_file train_data/QA/train.json \
    --validation_file train_data/QA/validation.json \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 5 \
    --source_lang "src" \
    --target_lang "tgt"
