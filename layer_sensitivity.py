import torch.nn as nn
from datasets import load_dataset
import os
import argparse
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from svd_utils import get_rank


def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        # Wrapper for tokenized input IDs
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    else:
        raise NotImplementedError


def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        # Wrapper for tokenized input IDs
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        # testenc = tokenizer("\n\n".join(valdata["text"]), return_tensors="pt")
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    else:
        raise NotImplementedError


@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, test_layer, seqlen=2048, device="cuda"):
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

    results = {}

    for dataset in datasets.split(","):
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader, weights_only=False)
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer)
            torch.save(testloader, cache_testloader)

        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        # model.config.use_cache = False
        model.eval()

        nlls = []

        for i in tqdm(range(10)):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(
                device
            )
            outputs = model.model(batch, test_layer=test_layer)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                           :, 1:
                           ].to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results.update({dataset: ppl.item()})

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None,
                        help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")

    parser.add_argument("--max_num_examples", type=int, default=None,
                        help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"],
                        help="how to sample the examples.")

    parser.add_argument("--max_new_tokens", type=int, default=None, help="")

    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")

    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--quant_method", type=str, default=None, choices=["kivi", "kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--rank", type=int, default=4096, help="rank of up and down matrix")
    parser.add_argument("--layer_step", type=int, default=2, help="how many layers connect to one")

    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )

    args = parser.parse_args()

    if args.quant_method == "kvquant":
        from commonkv.quantcache import KVQuantizedCache
        from transformers import cache_utils

        cache_utils.HQQQuantizedCache = KVQuantizedCache
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )

    import torch.nn.functional as F
    # extra_ppl = [133.20459365844727, 45.191802978515625, 21.15629005432129, 33.10429000854492, 8.482664108276367, 7.751203536987305,
    #  2.029176712036133, 1.3723411560058594, 0.6777997016906738, 1.3238582611083984, 0.6252565383911133,
    #  0.6459336280822754, 0.5848040580749512, 0.7922697067260742, 0.9161195755004883, 0.8141617774963379,
    #  0.6156749725341797, 0.6407599449157715, 1.0919160842895508, 0.9084649085998535, 1.062995433807373,
    #  0.809626579284668, 0.7187528610229492, 0.8467345237731934, 0.49149656295776367, 0.8573689460754395,
    #  0.2780289649963379, 0.6696343421936035, 0.45176076889038086, 0.5965499877929688, 0.3012847900390625,
    #  0.5745425224304199]
    # group_size = 4
    # for i in range(0, len(extra_ppl), group_size):
    #     group = extra_ppl[i:i + group_size]
    #     group_sum = sum(group)
    #     if group_sum == 0:
    #         normalized_group = [0.0] * group_size
    #     else:
    #         normalized_group = [x / group_sum for x in group]
    #     extra_ppl[i:i + group_size] = normalized_group
    # print(extra_ppl)

    from commonkv.monkeypatch import replace_llama, replace_mistral

    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation,
        rank=args.rank,
        layer_step=args.layer_step
    )

    if args.method == 'Ours':
        RANK = args.rank
        layer_step = args.layer_step
        num_layers = len(model.model.layers)
        head_wise_ranks = get_rank(args.model_path)

        for start_idx in tqdm(range(0, num_layers, layer_step)):
            end_idx = min(start_idx + layer_step, num_layers)
            layers = [model.model.layers[i].self_attn for i in range(start_idx, end_idx)]

            k_weights = [layer.k_proj.weight.data.float() for layer in layers]
            k_cat = torch.cat(k_weights, dim=0)
            v_weights = [layer.v_proj.weight.data.float() for layer in layers]
            v_cat = torch.cat(v_weights, dim=0)
            kv_cat = torch.cat([k_cat, v_cat], dim=0)

            KVU, KVS, KVVt = torch.linalg.svd(kv_cat, full_matrices=False)

            offset = 0
            for i, layer in enumerate(layers):
                current_layer_num = start_idx + i
                current_rank = head_wise_ranks[f'model.layers.{current_layer_num}.self_attn.k_proj'][0]

                KV_reduced = (KVU[:, :current_rank] @ torch.diag(KVS[:current_rank]))
                KVVt_reduced = KVVt[:current_rank]

                D_out = k_weights[i].shape[0]
                layer.k_down_proj.weight.data = KV_reduced[offset:offset + D_out].half()
                layer.k_up_proj.weight.data = KVVt_reduced.to(torch.float16)
                offset += D_out
                layer.k_proj = None
            for i, layer in enumerate(layers):
                current_layer_num = start_idx + i
                current_rank = head_wise_ranks[f'model.layers.{current_layer_num}.self_attn.v_proj'][0]

                KV_reduced = (KVU[:, :current_rank] @ torch.diag(KVS[:current_rank]))
                KVVt_reduced = KVVt[:current_rank]

                D_out = v_weights[i].shape[0]
                layer.v_down_proj.weight.data = KV_reduced[offset:offset + D_out].half()
                layer.v_up_proj.weight.data = KVVt_reduced.to(torch.float16)
                offset += D_out
                layer.v_proj = None

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    save_dir = args.save_dir
    max_capacity_prompts = args.max_capacity_prompts

    results = eval_ppl(model, tokenizer, "Llama3.1-8B-Instruct", "wikitext2", test_layer=33)
    fullkv_ppl = results['wikitext2']
    print("full kv ppl:")
    print(results)
    extra_ppl = []
    for i in range(32):
        results = eval_ppl(model, tokenizer, "Llama3.1-8B-Instruct", "wikitext2", test_layer=i)
        print(f"test_layer:{i}")
        print(results)
        extra_ppl.append(results['wikitext2']-fullkv_ppl)
    print(extra_ppl)

    group_size = 4
    for i in range(0, len(extra_ppl), group_size):
        group = torch.tensor(extra_ppl[i:i + group_size])
        norm_group = (group - group.min()) / (group.max() - group.min() + 1e-8)
        extra_ppl[i:i + group_size] = norm_group.tolist()
    print(extra_ppl)
