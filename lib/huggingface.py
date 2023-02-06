from transformers import AutoModelForCausalLM, AutoConfig


def get_model_huggingface(model_name_or_path: str) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(
        model_name_or_path, cache_dir=None, use_cache=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=None,
    )
    return model
