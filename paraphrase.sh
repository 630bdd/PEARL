export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/x86_64-linux-gnu/
export PATH=$PATH:$HOME/.local/bin
python paraphrase.py \
    --dataset CompQ \
    --model_path models/PEARL_generator_for_MPT-7B_on_QA \
    --batch_size 32 \
    --output_dir paraphrase