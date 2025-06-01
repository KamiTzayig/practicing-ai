

import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

"""## Loading a Quantized Base Model

We start by loading a quantized model, so it takes up less space in the GPU's RAM. A quantized model replaces the original weights with approximate values that are represented by fewer bits. The simplest and most straightforward way to quantize a model is to turn its weights from 32-bit floating-point (FP32) numbers into 4-bit floating-point numbers (NF4). This simple yet powerful change already **reduces the model's memory footprint** by roughly a factor of eight.

We can use an instance of `BitsAndBytesConfig` as the `quantization_config` argument while loading a model using the `from_pretrained()` method. To keep it flexible, so you can try it out with any other model of your choice, we're using Hugging Face's
`AutoModelForCausalLM`. The repo you choose to use determines the model being loaded.

Without further ado, here's our quantized model being loaded:
"""

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)
repo_id = 'meta-llama/Llama-3.2-1B-Instruct'
model = AutoModelForCausalLM.from_pretrained(repo_id,
                                             device_map="cuda:0",
                                             quantization_config=bnb_config
)

"""<blockquote class="note">
  <p>
    <em>"The Phi-3-Mini-4K-Instruct is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the Mini version in two variants 4K and 128K which is the context length (in tokens) that it can support."</em>
    <br>
    Source: <a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">Hugging Face Hub</a>
  </p>
</blockquote>

Once the model is loaded, you can see how much space it occupies in memory using the `get_memory_footprint()` method.
"""

print(model.get_memory_footprint()/1e6)

"""Even though it's been quantized, the model still takes up a bit more than 2 gigabytes of RAM. The **quantization** procedure focuses on the **linear layers within the Transformer decoder blocks** (also referred to as "layers" in some cases):"""

model

"""A **quantized model** can be used directly for inference, but it **cannot be trained any further**. Those pesky `Linear4bit` layers take up much less space, which is the whole point of quantization; however, we cannot update them.

We need to add something else to our mix, a sprinkle of adapters.

## Setting Up Low-Rank Adapters (LoRA)

Low-rank adapters can be attached to each and every one of the quantized layers. The **adapters** are mostly **regular `Linear`  layers** that can be easily updated as usual. The clever trick in this case is that these adapters are significantly **smaller** than the layers that have been quantized.

Since the **quantized layers are frozen** (they cannot be updated), setting up **LoRA adapters** on a quantized model drastically **reduces the total number of trainable parameters** to just 1% (or less) of its original size.

We can set up LoRA adapters in three easy steps:

* Call `prepare_model_for_kbit_training()` to _improve numerical stability_ during training.
* Create an instance of `LoraConfig`.
* Apply the configuration to the quantized base model using the `get_peft_model()` method.

Let's try it out with our model:
"""

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,                   # the rank of the adapter, the lower the fewer parameters you'll need to train
    lora_alpha=16,         # multiplier, usually 2*r
    bias="none",           # BEWARE: training biases *modifies* base model's behavior
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # Newer models, such as Phi-3 at time of writing, may require
    # manually setting target modules
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)

model = get_peft_model(model, config)
model

"""The output of the other three LoRA layers (`qkv_proj`, `gate_up_proj`, and `down_proj`) was suppressed to shorten the output.

<blockquote class="warning">
  <p>
    Did you get the following error?
    <br>
    <br>
    <tt>ValueError: Please specify `target_modules` in `peft_config`</tt>
    <br>
    <br>
    Most likely, you don't need to specify the <tt>target_modules</tt> if you're using one of the well-known models. The <tt>peft</tt> library takes care of it by <em>automatically choosing the appropriate targets</em>. However, there may be a gap between the time a popular model is released and the time the library gets updated. So, if you get the error above, look for the quantized layers in your model and list their names in the <tt>target_modules</tt> argument.
  </p>
</blockquote>

The quantized layers (`Linear4bit`) have turned into `lora.Linear4bit` modules where the quantized layer itself became the `base_layer` with some regular `Linear` layers (`lora_A` and `lora_B`) added to the mix.

These extra layers would make the model only slightly larger. However, **the model preparation function** (`prepare_model_for_kbit_training()`) turned **every non-quantized layer to full precision (FP32)**, thus resulting in a 30% larger model:
"""

print(model.get_memory_footprint()/1e6)

"""Since most parameters are frozen, only a tiny fraction of the total number of parameters are currently trainable, thanks to LoRA!"""

trainable_parms, tot_parms = model.get_nb_trainable_parameters()
print(f'Trainable parameters:             {trainable_parms/1e6:.2f}M')
print(f'Total parameters:                 {tot_parms/1e6:.2f}M')
print(f'Fraction of trainable parameters: {100*trainable_parms/tot_parms:.2f}%')

"""The model is ready to be fine-tuned, but we are still missing one key component: our dataset.

## Formatting Your Dataset

<blockquote style="quotes: none !important;">
  <p>
    <em>"Like Yoda, speak, you must. Hrmmm."</em>
    <br>
    <br>
    Master Yoda
  </p>
</blockquote>

The dataset [`yoda_sentences`](https://huggingface.co/datasets/dvgodoy/yoda_sentences) consists of 720 sentences translated from English to Yoda-speak. The dataset is hosted on the Hugging Face Hub and we can easily load it using the `load_dataset()` method from the Hugging Face `datasets` library:
"""

dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
dataset

"""The dataset has three columns:

* original English sentence (`sentence`)
* basic translation to Yoda-speak (`translation`)
* enhanced translation including typical `Yesss` and `Hrrmm` interjections (`translation_extra`)
"""

dataset[0]

"""The `SFTTrainer` we'll be using to fine-tune the model can automatically handle datasets either in **conversational** or **instruction** formats.

* **conversational format**

```
{"messages":[
  {"role": "system", "content": "<general directives>"},
  {"role": "user", "content": "<prompt text>"},
  {"role": "assistant", "content": "<ideal generated text>"}
]}
```

* **instruction format**: **[unfortunately, recent versions of `trl` do not support this format properly anymore, please check the important update a few cells below]**

```
{"prompt": "<prompt text>",
"completion": "<ideal generated text>"}
```

Since the instruction format is easier to work with, we'll simply rename and keep the relevant columns from our dataset. That's it for formatting.
"""

dataset = dataset.rename_column("sentence", "prompt")
dataset = dataset.rename_column("translation_extra", "completion")
dataset = dataset.remove_columns(["translation"])
dataset

dataset[0]

"""Internally, the training data will be converted from the instruction to the conversational format:"""

messages = [
    {"role": "user", "content": dataset[0]['prompt']},
    {"role": "assistant", "content": dataset[0]['completion']}
]
messages

"""***

**IMPORTANT UPDATE**: unfortunately, in more recent versions of the `trl` library, the "instruction" format is not properly supported anymore, thus leading to the chat template not being applied to the dataset. In order to avoid this issue, we can convert the dataset to the "conversational" format.

***
"""

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
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

dataset = dataset.map(format_dataset).remove_columns(['prompt', 'completion'])

"""### Tokenizer

Before moving into the actual training, we still need to **load the tokenizer that corresponds to our model**. The tokenizer is an important part of this process, determining how to convert text into tokens in the same way used to train the model.

For instruction/chat models, the tokenizer also contains its corresponding **chat template** that specifies:

* Which **special tokens** should be used, and where they should be placed.
* Where the system directives, user prompt, and model response should be placed.
* What is the **generation prompt**, that is, the special token that triggers the model's response (more on that in the "Querying the Model" section)
"""

tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.chat_template

"""Never mind the seemingly overcomplicated template (I have added line breaks and indentation to it so it's easier to read). It simply organizes the messages into a coherent block with the appropriate tags, as shown below (`tokenize=False` ensures we get readable text back instead of a numeric sequence of token IDs):"""

print(tokenizer.apply_chat_template(messages, tokenize=False))

"""Notice that each interaction is wrapped in either `<|user|>` or `<|assistant|>` tokens at the beginning and `<|end|>` at the end. Moreover, the `<|endoftext|>` token indicates the end of the whole block.

Different models will have different templates and tokens to indicate the beginning and end of sentences and blocks.

We're now ready to tackle the actual fine-tuning!

***
**IMPORTANT UPDATE**: due to changes in the default collator used by the `SFTTrainer` class while building the dataset, the EOS token (which is, in Phi-3, the same as the PAD token) was masked in the labels too thus leading to the model not being able to properly stop token generation.

In order to address this change, we can assign the UNK token to the PAD token, so the EOS token becomes unique and therefore not masked as part of the labels.
***
"""

# tokenizer.pad_token = tokenizer.unk_token
# tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.pad_token = tokenizer.eos_token

"""## Fine-Tuning with SFTTrainer

**Fine-tuning a model**, whether large or otherwise, follows exactly **the same training procedure as training a model from scratch**. We could write our own training loop in pure PyTorch, or we could use Hugging Face's `Trainer` to fine-tune our model.

It is much easier, however, to use `SFTTrainer` instead (which uses `Trainer` underneath, by the way), since it takes care of most of the nitty-gritty details for us, as long as we provide it with the following four arguments:

* a model
* a tokenizer
* a dataset
* a configuration object

We've already got the first three elements; let's work on the last one.

### SFTConfig

There are many parameters that we can set in the configuration object. We have divided them into four groups:

* **Memory usage** optimization parameters related to **gradient accumulation and checkpointing**
* **Dataset**-related arguments, such as the `max_seq_length` required by your data, and whether you are packing or not the sequences
* Typical **training parameters** such as the `learning_rate` and the `num_train_epochs`
* **Environment and logging** parameters such as `output_dir` (this will be the name of the model if you choose to push it to the Hugging Face Hub once it's trained), `logging_dir`, and `logging_steps`.

While the _learning rate_ is a very important parameter (as a starting point, you can try the learning rate used to train the base model in the first place), it's actually the **maximum sequence length** that's more likely to cause **out-of-memory issues**.

Make sure to always pick the shortest possible `max_seq_length` that makes sense for your use case. In ours, the sentences—both in English and Yoda-speak—are quite short, and a sequence of 64 tokens is more than enough to cover the prompt, the completion, and the added special tokens.

<blockquote class="tip">
  <p>
    Flash attention (which, unfortunately, isn't supported in Colab), allows for more flexibility in working with longer sequences, avoiding the potential issue of OOM errors.
  </p>
</blockquote>
"""

sft_config = SFTConfig(
    ## GROUP 1: Memory usage
    # These arguments will squeeze the most out of your GPU's RAM
    # Checkpointing
    gradient_checkpointing=True,
    # this saves a LOT of memory
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={'use_reentrant': False},
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=16,
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=True,

    ## GROUP 2: Dataset-related
    max_seq_length=64,
    # Dataset
    # packing a dataset means no padding is needed
    packing=True,

    ## GROUP 3: These are typical training parameters
    num_train_epochs=10,
    learning_rate=3e-4,
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim='paged_adamw_8bit',

    ## GROUP 4: Logging parameters
    logging_steps=10,
    logging_dir='./logs',
    output_dir='./phi3-mini-yoda-adapter',
    report_to='none'
)

"""### `SFTTrainer`

<blockquote style="quotes: none !important;">
  <p>
    <em>"It is training time!"</em>
    <br>
    <br>
    The Hulk
  </p>
</blockquote>

We can now finally create an instance of the supervised fine-tuning trainer:
"""

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset,
)

"""The `SFTTrainer` had already preprocessed our dataset, so we can take a look inside and see how each mini-batch was assembled:"""

dl = trainer.get_train_dataloader()
batch = next(iter(dl))

batch['input_ids'][0], batch['labels'][0]

"""The **labels were added automatically**, and they're **exactly the same as the inputs**. Thus, this is a case of **self-supervised fine-tuning**.

The shifting of the labels will be handled automatically as well; there's no need to be concerned about it.

<blockquote class="note">
  <p>
    Although this is a 3.8 billion-parameter model, the configuration above allows us to squeeze training, using a mini-batch of eight, into an old setup with a consumer-grade GPU such as a GTX 1060 with only 6 GB RAM. True story!
    <br>
    It takes about 35 minutes to complete the training process.
  </p>
</blockquote>

Next, we call the `train()` method and wait:
"""

trainer.train()

"""## Querying the Model

Now, our model should be able to produce a Yoda-like sentence as a response to any short sentence we give it.

So, the model requires its inputs to be properly formatted. We need to build a list of "messages"—ours, from the `user`, in this case—and prompt the model to answer by indicating it's its turn to write.

This is the purpose of the `add_generation_prompt` argument: it adds `<|assistant|>` to the end of the conversation, so the model can predict the next word—and continue doing so until it predicts an `<|endoftext|>` token.

The helper function below assembles a message (in the conversational format) and **applies the chat template** to it, **appending the generation prompt** to its end.
"""

def gen_prompt(tokenizer, sentence):
    converted_sample = [
        {"role": "user", "content": sentence},
    ]
    prompt = tokenizer.apply_chat_template(converted_sample,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt

"""Let's try generating a prompt for an example sentence:"""

sentence = 'you are one of my best friends!'
prompt = gen_prompt(tokenizer, sentence)
print(prompt)

"""The prompt seems about right; let's use it to generate a completion. The helper function below does the following:

* It **tokenizes the prompt** into a tensor of token IDs (`add_special_tokens` is set to `False` because the tokens were already added by the chat template).
* It sets the model to **evaluation mode**.
* It calls the model's `generate()` method to **produce the output** (generated token IDs).
* It **decodes the generated token IDs** back into readable text.
"""

def generate(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    tokenized_input = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    model.eval()
    generation_output = model.generate(**tokenized_input,
                                       eos_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=max_new_tokens)

    output = tokenizer.batch_decode(generation_output,
                                    skip_special_tokens=skip_special_tokens)
    return output[0]

"""Now, we can finally try out our model and see if it's indeed capable of generating Yoda-speak."""

print(generate(model, tokenizer, prompt))

"""Awesome! It works! Like Yoda, the model speaks. Hrrrmm.

Congratulations, you've fine-tuned your first LLM!

Now, you've got a small adapter that can be loaded into an instance of the Phi-3 Mini 4K Instruct model to turn it into a Yoda translator! How cool is that?

### Saving the Adapter

Once the training is completed, you can save the adapter (and the tokenizer) to disk by calling the trainer's `save_model()` method. It will save everything to the specified folder:
"""

trainer.save_model('local-phi3-mini-yoda-adapter')

"""The files that were saved include:

* the  adapter configuration (`adapter_config.json`) and weights (`adapter_model.safetensors`)—the adapter itself is just 50 MB in size
* the training arguments (`training_args.bin`)
* the tokenizer (`tokenizer.json` and `tokenizer.model`), its configuration (`tokenizer_config.json`), and its special tokens (`added_tokens.json` and `speciak_tokens_map.json`)
* a README file
"""

os.listdir('local-phi3-mini-yoda-adapter')

"""If you'd like to share your adapter with everyone, you can also push it to the Hugging Face Hub. First, log in using a token that has permission to write:"""

from huggingface_hub import login
login()

"""The code above will ask you to enter an access token:

![](https://github.com/dvgodoy/FineTuningLLMs/blob/main/images/ch0/hub0.png?raw=True)
<center>Figure 0.1 - Logging into the Hugging Face Hub</center>

A successful login should look like this (pay attention to the permissions):

![](https://github.com/dvgodoy/FineTuningLLMs/blob/main/images/ch0/hub1.png?raw=True)
<center>Figure 0.2 - Successful Login</center>

Then, you can use the trainer's `push_to_hub()` method to upload everything to your account in the Hub. The model will be named after the `output_dir` argument of the training arguments:
"""

trainer.push_to_hub()

"""There you go! Our model is out there in the world, and anyone can use it to translate English into Yoda speak.

That's a wrap!
"""