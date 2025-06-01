import torch

# Adapted from trl.extras.dataset_formatting.instructions_formatting_function
# Converts dataset from prompt/completion format (not supported anymore)
# to the conversational format
def format_dataset(examples):
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {"messages": output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {"messages": converted_sample}

def gen_prompt(tokenizer, sentence):
    converted_sample = [{"role": "user", "content": sentence}]
    prompt = tokenizer.apply_chat_template(
        converted_sample, tokenize=False, add_generation_prompt=True
    )
    return prompt

def generate_text(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    tokenized_input = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    model.eval()
    gen_output = model.generate(**tokenized_input,
                                eos_token_id=tokenizer.eos_token_id,
                                max_new_tokens=max_new_tokens)
    
    output = tokenizer.batch_decode(gen_output, skip_special_tokens=skip_special_tokens)
    return output[0] 