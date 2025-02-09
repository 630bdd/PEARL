export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/x86_64-linux-gnu/
export PATH=$PATH:$HOME/.local/bin
python inference.py \
    --dataset CompQ \
    --model_path models/MPT-instruct-7B \
    --epr