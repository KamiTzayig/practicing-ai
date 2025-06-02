# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="llama-3.2-1b-yoda-adapter-cpu")
messages = [
    {"role": "user", "content": "you are one of my best friends"},
]
print(pipe(messages))
