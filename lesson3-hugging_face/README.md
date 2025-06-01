# Fine-Tuning LLMs with Hugging Face

This directory contains scripts to fine-tune a Large Language Model (LLM) using Hugging Face libraries, based on the [Fine-Tuning Your First Large Language Model (LLM) with PyTorch and Hugging Face](https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face) blog post.

## Setup

1.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate 
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Login to Hugging Face Hub**:
    If you want to push your fine-tuned adapter to the Hugging Face Hub, you'll need to log in.
    ```bash
    huggingface-cli login
    ```
    Alternatively, you can provide your token directly in the script or as an environment variable.

## Running the Fine-Tuning Script

The main script for fine-tuning is `fine_tune.py`.

```bash
python fine_tune.py
```

This script will:
1.  Load the base model (`microsoft/Phi-3-mini-4k-instruct` by default) and quantize it.
2.  Set up Low-Rank Adaptation (LoRA).
3.  Load the dataset (`dvgodoy/yoda_sentences` by default) and format it for conversational fine-tuning.
4.  Configure the `SFTTrainer`.
5.  Train the model.
6.  Save the fine-tuned adapter locally (in a directory named `phi3-mini-yoda-adapter-local` by default).
7.  Run a quick test with a sample sentence to see the fine-tuned model in action.

## Configuration

You can modify the following variables at the beginning of `fine_tune.py` to change the behavior:

*   `MODEL_REPO_ID`: The Hugging Face Hub repository ID of the base model to fine-tune.
*   `DATASET_ID`: The Hugging Face Hub ID of the dataset to use for fine-tuning.
*   `NEW_MODEL_NAME`: The local directory name where the fine-tuned adapter will be saved.
*   `HF_HUB_MODEL_NAME`: The name for your model on the Hugging Face Hub if you choose to upload it (remember to include your username, e.g., `YourUsername/phi3-mini-yoda-adapter`).

The section for pushing the model to the Hub is commented out by default in `fine_tune.py`. Uncomment it and set `HF_HUB_MODEL_NAME` if you wish to use this feature.

## Scripts

*   `fine_tune.py`: The main script that orchestrates the fine-tuning process.
*   `utils.py`: Contains helper functions for dataset formatting and text generation.
*   `requirements.txt`: Lists the Python dependencies.
*   `test_llama_1b.py`: Your original script for testing a Llama model (not directly used by the fine-tuning process).

## Notes
* This setup is designed to be relatively memory-efficient, utilizing 4-bit quantization and LoRA. However, fine-tuning LLMs can still be resource-intensive.
* The `SFTConfig` in `fine_tune.py` includes `auto_find_batch_size=True`, which can help prevent out-of-memory errors by automatically adjusting the batch size.
* The default dataset and task is to make the model speak like Yoda. You can adapt this to other datasets and tasks by changing `DATASET_ID` and potentially modifying the `format_dataset` function in `utils.py` if your dataset has a different structure. 