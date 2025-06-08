# Google Colab Fine-tuning Script
#
# 1. Make sure to select a GPU runtime in Colab: Runtime -> Change runtime type -> GPU (T4, A100, etc.)
# 2. Install necessary libraries by running this cell in Colab:
#    !pip install -q torch # PyTorch should usually be pre-installed
#    !pip install -q transformers datasets peft trl bitsandbytes accelerate huggingface_hub
#
# 3. For pushing the model to Hugging Face Hub, you'll need to log in.
#    You can do this by running in a Colab cell:
#    from huggingface_hub import notebook_login
#    notebook_login()
#    Or by setting your token as an environment variable (less secure for notebooks):
#    import os
#    os.environ["HF_TOKEN"] = "YOUR_HUGGINGFACE_TOKEN_HERE"
#
# This script is adapted for GPU usage with 4-bit quantization.

import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login # For programmatic login, or use notebook_login()

# --- Inlined utility functions from utils.py ---
def gen_prompt(tokenizer, sentence):
    converted_sample = [{"role": "user", "content": sentence}]
    prompt = tokenizer.apply_chat_template(
        converted_sample, tokenize=False, add_generation_prompt=True
    )
    return prompt

def generate_text(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    tokenized_input = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device) # Ensure input is on the same device as the model

    model.eval()
    with torch.no_grad(): # Important for inference
        gen_output = model.generate(**tokenized_input,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id, # Explicitly set pad_token_id
                                    max_new_tokens=max_new_tokens)
    
    output = tokenizer.batch_decode(gen_output, skip_special_tokens=skip_special_tokens)
    return output[0]
# --- End of inlined utility functions ---

MODEL_REPO_ID = 'meta-llama/Llama-3.2-1B-Instruct' # Or any other model
DATASET_ID = "NousResearch/hermes-function-calling-v1"
DATASET_NAME = "func_calling" # Specific subset of the Hermes dataset
NEW_MODEL_NAME_COLAB = "llama-3.2-1b-hermes-fc-adapter-colab"
# Replace with your actual HF username and desired model name for the adapter
HF_HUB_MODEL_NAME = "KamiTzayig/Llama-3.2-1B-hermes-fc-adapter"


def load_model_and_tokenizer_gpu(repo_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPUs (T4, A100)
        # bnb_4bit_use_double_quant=True, # Optional: can improve results slightly
    )
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=quantization_config,
        device_map="auto", # Automatically distribute model on available GPUs/CPU
        # torch_dtype=torch.bfloat16, # Can also set here, but BitsAndBytesConfig handles compute_dtype
        trust_remote_code=True # Required for some models like Phi-3 if not using official HF implementation
    )
    
    # Prepares the model for k-bit training (e.g., QLoRA)
    # This includes settings like enabling gradient checkpointing and ensuring proper casting
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    # Pad token handling:
    # The `load_and_format_dataset` function (copied from your fine_tune.py)
    # will set tokenizer.pad_token = tokenizer.unk_token.
    # Here, we ensure that if pad_token is somehow still None after AutoTokenizer,
    # it defaults to eos_token before `load_and_format_dataset` is called.
    # `load_and_format_dataset`'s logic will then take precedence if it finds pad_token to be eos.
    if tokenizer.pad_token is None:
        print(f"tokenizer.pad_token was None. Setting pad_token = eos_token ({tokenizer.eos_token}).")
        tokenizer.pad_token = tokenizer.eos_token
    
    # SFTTrainer needs tokenizer.pad_token_id to be set.
    # model.config.pad_token_id is also important for generation config.
    if tokenizer.pad_token_id is None: # Should be set if pad_token was set to eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.pad_token_id = tokenizer.pad_token_id # Synchronize model config

    print(f"Tokenizer after init: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}, eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}")
    if hasattr(model, 'get_memory_footprint'):
        print(f"Model memory footprint: {model.get_memory_footprint()/1e6:.2f} MB")
    return model, tokenizer

def setup_lora(model):
    lora_config = LoraConfig(
        r=16, # Increased rank for potentially better performance, adjust as needed
        lora_alpha=32, # alpha = 2*r is a common rule of thumb
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=['o_proj', 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'], # Expanded target modules for Llama-3
    )
    model = get_peft_model(model, lora_config)
    train_p, tot_p = model.get_nb_trainable_parameters()
    print(f"Trainable LoRA parameters: {train_p/1e6:.2f}M")
    print(f"Total parameters: {tot_p/1e6:.2f}M")
    print(f"% of trainable parameters: {100*train_p/tot_p:.2f}%")
    return model

# This function is taken from your modified fine_tune.py
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
                elif role == 'system': # System messages are usually kept as 'system'
                    pass
                elif role == 'tool': # Explicitly handle 'tool' role
                    pass # Keep role as 'tool' for apply_chat_template
                else: # Fallback for unknown roles
                    print(f"Warning: Unhandled role '{role}' found. Mapping to 'user'. Adjust if needed.")
                    role = 'user'
                current_chat_turns.append({'role': role, 'content': turn['value']})
            
            try:
                # For training, we don't add a generation prompt typically.
                # The trainer handles labels based on the sequence.
                formatted_chat_string = tokenizer.apply_chat_template(
                    current_chat_turns, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                all_formatted_chats.append(formatted_chat_string)
            except Exception as e:
                print(f"Error applying chat template to: {current_chat_turns}")
                print(f"Error: {e}")
                all_formatted_chats.append("") # Add a placeholder or skip
        return {'text': all_formatted_chats}

    dataset = dataset.map(
        transform_conversations_for_hermes,
        batched=True,
        remove_columns=dataset.column_names 
    )
    
    # The pad token should have been set in load_model_and_tokenizer_gpu.
    # This section now just confirms and prints. The complex unk_token logic is removed.
    if tokenizer.pad_token is None:
        # This case should ideally not be reached if load_model_and_tokenizer_gpu did its job
        print("CRITICAL WARNING in load_and_format_dataset: tokenizer.pad_token is STILL None. Defaulting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.pad_token_id is None:
        # This case should ideally not be reached
        print(f"CRITICAL WARNING in load_and_format_dataset: tokenizer.pad_token_id is STILL None. Defaulting to eos_token_id ({tokenizer.eos_token_id}).")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer final check in load_and_format_dataset: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")
    print(f"Tokenizer final check in load_and_format_dataset: eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}")

    return dataset

def get_sft_config(output_dir):
    return SFTConfig(
        # Optimizations for GPU
        gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={'use_reentrant': False}, # Often default, can be explicit
        bf16=True, # Enable bfloat16 mixed-precision training (requires Ampere GPU or newer)
                   # If on older GPUs (e.g., K80, P100), set fp16=True instead.
        
        # Batching and Steps
        per_device_train_batch_size=2,  # Start small for Colab T4, auto_find_batch_size will try to increase
        auto_find_batch_size=True,      # Automatically finds the largest batch size that fits
        gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
        
        max_seq_length=512, # Increased sequence length; adjust based on task and memory
                            # Original was 64. Hermes func calling might have longer sequences.
        packing=True,              # Efficiently packs multiple short examples into one sequence
        dataset_text_field="text", # Column in dataset containing the text to train on
        
        # Training Hyperparameters
        num_train_epochs=3,        # Adjust as needed; 3 is a good start for Colab
        learning_rate=2e-4,        # Common learning rate for LoRA
        lr_scheduler_type="cosine", # Learning rate scheduler
        warmup_ratio=0.03,         # Warmup steps for learning rate
        weight_decay=0.001,        # Weight decay
        optim='paged_adamw_8bit',  # Memory-efficient optimizer for quantized models
                                   # or 'adamw_torch_fused' if paged optimizer gives issues/not preferred

        # Logging and Output
        logging_steps=25,           # Log training progress more frequently
        logging_dir=f'./{output_dir}_logs', # Separate log directory
        output_dir=output_dir,
        report_to='none', # Set to 'wandb', 'tensorboard', or 'all' if you use them

        # Saving strategy
        save_strategy="epoch",    # Save a checkpoint at the end of each epoch
        save_total_limit=1,       # Keep only the best or last checkpoint
        
        # Seed for reproducibility
        seed=42,
    )

def train_model(model, tokenizer, dataset, sft_config):
    print(f"Tokenizer PAD token ID before SFTTrainer: {tokenizer.pad_token_id}")
    print(f"Tokenizer EOS token ID before SFTTrainer: {tokenizer.eos_token_id}")

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None before starting SFTTrainer. This must be set.")
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer, # Using processing_class as per older TRL or specific versions
        args=sft_config,
        train_dataset=dataset,
        # max_seq_length=sft_config.max_seq_length, # Already in SFTConfig
        # dataset_text_field=sft_config.dataset_text_field, # Already in SFTConfig
        # packing=sft_config.packing # Already in SFTConfig
    )
    trainer.train()
    return trainer

def main():
    # Optional: Set HuggingFace Hub token if needed for private models or pushing
    # from huggingface_hub import HfFolder
    # HfFolder.save_token('YOUR_HF_TOKEN') # If you prefer to save it this way

    # 1. Load model and tokenizer with quantization for GPU
    print("Loading model and tokenizer for GPU with quantization...")
    model, tokenizer = load_model_and_tokenizer_gpu(MODEL_REPO_ID)

    # 2. Setup LoRA
    print("Setting up LoRA...")
    model = setup_lora(model)

    # 3. Load and format dataset
    # The tokenizer is modified within load_and_format_dataset (pad token)
    print("Loading and formatting dataset...")
    dataset = load_and_format_dataset(DATASET_ID, tokenizer, name=DATASET_NAME)
    print(dataset)
    if len(dataset) > 0:
        print("Sample processed text from dataset:")
        print(dataset[0]['text'][:500] + "..." if dataset[0]['text'] else "N/A") # Print first 500 chars of first example
    else:
        print("Dataset is empty after processing.")
        return


    # 4. Configure SFTTrainer
    print("Configuring SFTTrainer...")
    sft_config = get_sft_config(output_dir=NEW_MODEL_NAME_COLAB)

    # 5. Train the model
    print("Starting training...")
    trainer = train_model(model, tokenizer, dataset, sft_config)
    print("Training finished.")

    # 6. Save the adapter
    print(f"Saving adapter to {NEW_MODEL_NAME_COLAB}...")
    trainer.save_model(NEW_MODEL_NAME_COLAB) # Saves LoRA adapter
    # tokenizer.save_pretrained(NEW_MODEL_NAME_COLAB) # Optionally save tokenizer with adapter
    print("Adapter saved.")

    # 7. Test the fine-tuned model (optional)
    print("Testing the fine-tuned model (inference)...")
    test_sentence = "Can you write a Python function to calculate the factorial of a number?"
    # For function calling, test with a prompt that might elicit a function call
    # test_sentence = "What's the weather like in London?" # Example if model was trained for weather tool
    
    prompt = gen_prompt(tokenizer, test_sentence)
    print(f"Test Prompt: {prompt}")
    
    # Ensure the model used for generation is the trained one (trainer.model)
    # If LoRA, the trainer.model is the PeftModel
    generated_response = generate_text(trainer.model, tokenizer, prompt, max_new_tokens=150)
    print(f"Generated Response: {generated_response}")

    # 8. Push to Hub (optional)
    # Make sure you've run `notebook_login()` or set HF_TOKEN environment variable
    # Also, ensure HF_HUB_MODEL_NAME is correctly set to "your-username/your-repo-name"
    # print(f"Pushing adapter to Hugging Face Hub as {HF_HUB_MODEL_NAME}...")
    # try:
    #     # For programmatic login (if notebook_login() was not used or token set via env var):
    #     # from huggingface_hub import login
    #     # login(token="YOUR_HF_WRITE_TOKEN") # Replace with your write token if needed
    #
    #     trainer.push_to_hub(HF_HUB_MODEL_NAME)
    #     print(f"Adapter pushed to Hub: https://huggingface.co/{HF_HUB_MODEL_NAME}")
    # except Exception as e:
    #     print(f"Error pushing to hub: {e}")
    #     print("Make sure you are logged in (e.g. `from huggingface_hub import notebook_login; notebook_login()`)")
    #     print(f"And that your HF_HUB_MODEL_NAME ('{HF_HUB_MODEL_NAME}') is valid (e.g. 'YourUsername/YourModelName').")


if __name__ == "__main__":
    # For Colab, you'd typically run cells one by one, or call main() in a cell.
    # If running as a .py script:
    main() 