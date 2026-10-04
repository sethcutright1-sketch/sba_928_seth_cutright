"""this is a helper file for loading qwen and gerating answers for the step files """

from __future__ import annotations

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 250
def load_model(model_name):
    """Load the model and tokenizer for the given model name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    return model, tokenizer


def load_tuned_model(model_name, adapter_dir):
    """Load the tuned model with the specified adapter."""
    model, tokenizer = load_model(model_name)
    model = PeftModel.from_pretrained(model, adapter_dir)
    return model, tokenizer


def generate_reply(model, tokenizer, messages):
    """Generate a reply from the model based on the provided messages."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)