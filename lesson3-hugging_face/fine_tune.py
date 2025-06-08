import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login

from utils import format_dataset, gen_prompt, generate_text

MODEL_REPO_ID = 'meta-llama/Llama-3.2-1B-Instruct'
DATASET_ID = "NousResearch/hermes-function-calling-v1"
DATASET_NAME = "func_calling"
NEW_MODEL_NAME = "llama-3.2-1b-hermes-function-calling-adapter-cpu"
HF_HUB_MODEL_NAME = "phi3-mini-yoda-adapter" # Replace with your Hub username/model_name if you want to push

def load_model_on_cpu(repo_id):
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, device_map="cpu"
    )
    print(f"Model memory footprint: {model.get_memory_footprint()/1e6:.2f} MB (running on CPU)")
    return model

def setup_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_config)
    train_p, tot_p = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {train_p/1e6:.2f}M")
    print(f"Total parameters: {tot_p/1e6:.2f}M")
    print(f"% of trainable parameters: {100*train_p/tot_p:.2f}%")
    return model

def load_and_format_dataset(dataset_id, tokenizer, name=None):
    dataset = load_dataset(dataset_id, name=name, split="train")

    def transform_conversations_for_hermes(examples):
        all_formatted_chats = []
        for conversation_list in examples['conversations']:
            current_chat_turns = []
            for turn in conversation_list:
                role = turn['from']
                if role == 'human':
                    role = 'user'
                elif role == 'gpt':
                    role = 'assistant'
                elif role == 'system':
                    pass # System role is fine as is
                else:
                    # Assign a default role or handle other specific roles if necessary
                    # For SFTTrainer, 'user', 'assistant', 'system' are standard.
                    # If other roles appear and cause issues, they might need specific mapping.
                    print(f"Warning: Unhandled role '{role}' found. Mapping to 'user'. Adjust if needed.")
                    role = 'user' 
                current_chat_turns.append({'role': role, 'content': turn['value']})
            
            try:
                formatted_chat_string = tokenizer.apply_chat_template(
                    current_chat_turns, 
                    tokenize=False, 
                    add_generation_prompt=False # For training, generation prompt is typically not added here
                )
                all_formatted_chats.append(formatted_chat_string)
            except Exception as e:
                print(f"Error applying chat template to: {current_chat_turns}")
                print(f"Error: {e}")
                # Add a placeholder or skip this example if template application fails
                all_formatted_chats.append("") # Or handle error appropriately

        # The SFTTrainer with packing=True and dataset_text_field="text" expects a column named "text"
        return {'text': all_formatted_chats}

    # Apply the transformation.
    # Keep only the new 'text' column and remove all original columns.
    dataset = dataset.map(
        transform_conversations_for_hermes,
        batched=True,
        remove_columns=dataset.column_names 
    )
    
    # IMPORTANT UPDATE from the blog (or similar context): 
    # This PAD token logic is generally useful, especially for models like Phi-3.
    # It ensures EOS token is unique and not masked during label creation.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.unk_token
        # Ensure unk_token_id is not None. If it is, this might need a different strategy.
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            # Fallback: if unk_token is not set, this could be an issue.
            # For Llama-3, pad_token is often not set by default. Setting it to eos_token is common,
            # but the original comment suggests avoiding this for Phi-3.
            # Let's check if eos_token itself can be used if unk is not available,
            # or if another special token should be designated.
            # For robust behavior, if unk_token_id is None, we might need to add a new pad token.
            # However, most tokenizers have an unk_token.
            print("Warning: tokenizer.unk_token_id is None. pad_token setup might be incomplete.")


    return dataset

def get_sft_config(output_dir):
    return SFTConfig(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=1,
        per_device_train_batch_size=16,
        auto_find_batch_size=True,
        max_seq_length=64,
        packing=True,
        dataset_text_field="text",
        num_train_epochs=10,
        learning_rate=3e-4,
        optim='adamw_torch',
        logging_steps=10,
        logging_dir='./logs',
        output_dir=output_dir,
        report_to='none' # Set to 'wandb' or 'tensorboard' if you use them
    )

def train_model(model, tokenizer, dataset, sft_config):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
    )
    trainer.train()
    return trainer

def main():
    # 1. Load and quantize model
    print("Loading and quantizing model...")
    model = load_model_on_cpu(MODEL_REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

    # 2. Setup LoRA
    print("\nSetting up LoRA...")
    model = setup_lora(model)

    # 3. Load and format dataset
    print("\nLoading and formatting dataset...")
    dataset = load_and_format_dataset(DATASET_ID, tokenizer, DATASET_NAME,)
    print(dataset)
    print(dataset[0]['text'])

    # 4. Configure SFTTrainer
    print("\nConfiguring SFTTrainer...")
    sft_config = get_sft_config(output_dir=NEW_MODEL_NAME)

    # 5. Train the model
    print("\nStarting training...")
    trainer = train_model(model, tokenizer, dataset, sft_config)
    print("Training finished.")

    # 6. Save the adapter
    print(f"\nSaving adapter to {NEW_MODEL_NAME}...")
    trainer.save_model(NEW_MODEL_NAME)
    print("Adapter saved.")

    # 7. Test the fine-tuned model (optional)
    print("\nTesting the fine-tuned model...")
    test_sentence = 'The Force is strong in you!'
    prompt = gen_prompt(tokenizer, test_sentence)
    print(f"Prompt: {prompt}")
    generated_yoda_speak = generate_text(model, tokenizer, prompt)
    print(f"Generated: {generated_yoda_speak}")

    # 8. Push to Hub (optional)
    # print(f"\nPushing adapter to Hugging Face Hub as {HF_HUB_MODEL_NAME}...")
    # try:
    #     login() # Make sure you are logged in with `huggingface-cli login` or by providing a token
    #     trainer.push_to_hub(HF_HUB_MODEL_NAME)
    #     print("Adapter pushed to Hub.")
    # except Exception as e:
    #     print(f"Error pushing to hub: {e}")

if __name__ == "__main__":
    main() 