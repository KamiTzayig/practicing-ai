import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login

from utils import format_dataset, gen_prompt, generate_text

MODEL_REPO_ID = 'meta-llama/Llama-3.2-1B-Instruct'
DATASET_ID = "dvgodoy/yoda_sentences"
NEW_MODEL_NAME = "llama-3.2-1b-yoda-adapter-cpu-v2"
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

def load_and_format_dataset(dataset_id, tokenizer):
    dataset = load_dataset(dataset_id, split="train")
    dataset = dataset.rename_column("sentence", "prompt")
    dataset = dataset.rename_column("translation_extra", "completion")
    dataset = dataset.remove_columns(["translation"])
    dataset = dataset.map(format_dataset, batched=True).remove_columns(['prompt', 'completion'])
    
    # IMPORTANT UPDATE from the blog: 
    # due to changes in the default collator used by the SFTTrainer class while building the dataset, 
    # the EOS token (which is, in Phi-3, the same as the PAD token) was masked in the labels too 
    # thus leading to the model not being able to properly stop token generation.
    # In order to address this change, we can assign the UNK token to the PAD token, 
    # so the EOS token becomes unique and therefore not masked as part of the labels.
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    
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
    dataset = load_and_format_dataset(DATASET_ID, tokenizer)
    print(dataset)
    print(dataset[0]['messages'])

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