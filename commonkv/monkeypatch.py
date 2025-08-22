from importlib.metadata import version
import transformers

# from modeling_llama_svd import LlamaForCausalLM
from modeling_llama_svd_merge import LlamaForCausalLM
from modeling_mistral_svd_merge import MistralForCausalLM



from llama_model import prepare_inputs_for_generation_llama_new
from mistral_model import prepare_inputs_for_generation_mistral_new


def replace_llama(method, model_name=None):

    if method == "commonkv":
        print("Using CommonKV!")
        transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new

    


def replace_mistral(method):

    if method == "ours":
        print("Using ours!")
        transformers.models.mistral.modeling_mistral.MistralForCausalLM = MistralForCausalLM

    
    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new
