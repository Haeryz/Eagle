import re

import torch


def apply_chat_template(processor, messages):
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if hasattr(processor, "py_apply_chat_template"):
        return processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raise AttributeError("Processor does not provide a chat template API")


def process_vision_info(processor, messages):
    if hasattr(processor, "process_vision_info"):
        return processor.process_vision_info(messages)

    raise AttributeError("Processor does not provide process_vision_info")


def _maybe_to_device(value, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    return value


def prepare_generation_inputs(processor_inputs, device):
    return {
        "input_ids": processor_inputs["input_ids"].to(device),
        "attention_mask": _maybe_to_device(processor_inputs.get("attention_mask"), device),
        "pixel_values": _maybe_to_device(processor_inputs.get("pixel_values"), device),
        "image_grid_hws": _maybe_to_device(processor_inputs.get("image_grid_hws"), device),
    }


def build_generate_kwargs(prepared_inputs, processor, generation_mode, max_new_tokens, include_eos_token=False,
                          verbose=False, constrained_block=False, temperature=0.7, log_block_features=False):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "batch_decode"):
        tokenizer = processor

    generate_kwargs = dict(
        pixel_values=prepared_inputs["pixel_values"],
        input_ids=prepared_inputs["input_ids"],
        attention_mask=prepared_inputs["attention_mask"],
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        generation_mode=generation_mode,
        verbose=verbose,
        constrained_block=constrained_block,
        log_block_features=log_block_features,
    )

    if prepared_inputs["image_grid_hws"] is not None:
        generate_kwargs["image_grid_hws"] = prepared_inputs["image_grid_hws"]

    if include_eos_token and tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if generation_mode in ("fast", "hybrid"):
        generate_kwargs["n_future_tokens"] = 6

    return generate_kwargs


def decode_generation_output(raw_output, input_ids, processor):
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, list):
        if not raw_output:
            return ""
        if isinstance(raw_output[0], str):
            return raw_output[0]

    if torch.is_tensor(raw_output):
        generated_ids = raw_output
        if generated_ids.ndim == 2 and input_ids is not None and generated_ids.shape[1] >= input_ids.shape[1]:
            generated_ids = generated_ids[:, input_ids.shape[1]:]
        generated_ids = generated_ids.detach().cpu()

        if hasattr(processor, "post_process_image_text_to_text"):
            decoded = processor.post_process_image_text_to_text(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if isinstance(decoded, list):
                return decoded[0]
            return decoded

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None and hasattr(processor, "batch_decode"):
            tokenizer = processor
        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if isinstance(decoded, list) else decoded

    return str(raw_output)


def parse_statistic_info(raw_output):
    """Return the numeric fields of the 'Statistic Info' line emitted by generate(verbose=True), or None."""
    if not (isinstance(raw_output, tuple) and len(raw_output) == 3 and isinstance(raw_output[2], str)):
        return None
    stats = {}
    info, _, block_features = raw_output[2].partition("BlockFeatures=")
    if block_features:
        import json
        stats["blocks"] = json.loads(block_features)
    for key, value in re.findall(r"([\w\(\)]+)=([^;]+)", info):
        try:
            stats[key.strip()] = float(value)
        except ValueError:
            pass
    return stats


def load_model(model_path, dtype="bfloat16", lora_path=None):
    """Load LocateAnything through remote code, optionally in fp16 (pre-Ampere GPUs) and with a LoRA adapter."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=getattr(torch, dtype))
    if lora_path:
        from peft import PeftModel

        # comma-separated adapters are merged in order (e.g. SFT adapter, then the RL adapter trained on top of it)
        for path in lora_path.split(","):
            model.language_model = PeftModel.from_pretrained(model.language_model, path).merge_and_unload()
    return model
