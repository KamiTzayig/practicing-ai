# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
messages = [
    {"role": "user", "content": "where is maagan michael?"},
]
print(pipe(messages))