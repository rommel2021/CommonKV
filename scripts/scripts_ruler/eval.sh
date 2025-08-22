export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support CommonKV, FullKV
max_capacity_prompts=7950 # 128,2048,4096
attn_implementation=eager # Support  "eager".
result_path=$3  # path for saving results
model_path=$4  # model path
quant_method=$5 # Support kivi and kvquant, default None.
nbits=$6 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${result_path}"results_ruler" # path to result save_dir
rank=4096
layer_step=4

python3 run_ruler.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --layer_step ${layer_step} \
    --rank ${rank} \
    --save_dir ${save_dir} \
    --use_cache True \
#    --nbits ${nbits} \
#    --quant_method ${quant_method}
