export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support FullKV, CommonKV
max_capacity_prompts=8950 # 128,2048 in paper
attn_implementation=eager # Support "eager".
source_path=$3  # path for saving results
model_path=$4  # model path
quant_method=$5 # Support kivi and kvquant, default None.
nbits=$6 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${source_path}"results_long_bench" # path to result save_dir
rank=4098
layer_step=4

python run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --layer_step ${layer_step} \
    --rank ${rank} \
    --save_dir ${save_dir} \
#    --nbits ${nbits} \
#    --quant_method ${quant_method}
