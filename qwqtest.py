import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4",
    llm_int8_enable_fp32_cpu_offload = True
)

model_id = "Qwen/QwQ-32B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)

custom_device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    "model.layers.6": 0,
    "model.layers.7": 0,
    "model.layers.8": 0,
    "model.layers.9": 0,
    "model.layers.10": 0,
    "model.layers.11": 0,
    "model.layers.12": 0,
    "model.layers.13": 0,
    "model.layers.14": 0,
    "model.layers.15": 0,
    "model.layers.16": 0,
    "model.layers.17": 0,
    "model.layers.18": 0,
    "model.layers.19": 0,
    "model.layers.20": 0,
    "model.layers.21": 0,
    "model.layers.22": 0,
    "model.layers.23": 0,
    "model.layers.24": 0,
    "model.layers.25": 0,
    "model.layers.26": 0,
    "model.layers.27": 0,
    "model.layers.28": 0,
    "model.layers.29": 0,
    "model.layers.30": 0,
    "model.layers.31": 0,
    "model.layers.32": 0,
    "model.layers.33": 0,
    "model.layers.34": 0,
    "model.layers.35": 0,
    "model.layers.36": 0,
    "model.layers.37": 0,
    "model.layers.38": 0,
    "model.layers.39": 0,
    "model.layers.40": 0,
    "model.layers.41": 0,
    "model.layers.42": 0,
    "model.layers.43": 0,
    "model.layers.44": 0,
    "model.layers.45": 0,
    "model.layers.46": 0,
    "model.layers.47": 0,
    "model.layers.48": 0,
    "model.layers.49.self_attn": 0,
    "model.layers.49.mlp.gate_proj": 0,
    "model.layers.49.mlp.up_proj": 0,
    "model.layers.49.mlp.down_proj": 0,
    "model.layers.49.mlp.act_fn": 0,
    "model.layers.49.input_layernorm": 0,
    "model.layers.49.post_attention_layernorm": 0,
    "model.layers.50": 0,
    "model.layers.51": 0,
    "model.layers.52": 0,
    "model.layers.53": "cpu",
    "model.layers.54": "cpu",
    "model.layers.55": "cpu",
    "model.layers.56": "cpu",
    "model.layers.57": "cpu",
    "model.layers.58": "cpu",
    "model.layers.59": "cpu",
    "model.layers.60": "cpu",
    "model.layers.61": "cpu",
    "model.layers.62": "cpu",
    "model.layers.63": "cpu",
    "model.norm": "cpu", 
    "lm_head": "cpu"  
}

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=custom_device_map, 
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )


prompt = "How many times does the letter r appear in the word strawberry?"

messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")  

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
