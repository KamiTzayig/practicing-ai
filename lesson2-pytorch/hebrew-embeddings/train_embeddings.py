import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

train_file = "train.json"
vocabulary_file = "vocabulary.json"

with open(vocabulary_file, 'r', encoding='utf-8') as f:
    word_to_ix = json.load(f)
    ix_to_word = {v: k for k, v in word_to_ix.items()}
vocab_size = len(word_to_ix)

with open(train_file, 'r', encoding='utf-8') as f:
    raw_string = f.read()
    entries = raw_string.split("}\n")
    train_data = [json.loads(entry.rstrip() + "}") for entry in entries if entry.strip()]


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

embedding_dim = 500
learning_rate = 0.0001
epochs = 10
window_size = 2

model = EmbeddingModel(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

print("Starting training...")

# Initialize lists for plotting
plot_steps = []
plot_losses = []
cumulative_processed_pairs_for_plot = 0 # For cumulative plotting

# Setup interactive plot
plt.ion() # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot(plot_steps, plot_losses) # Initial empty plot
ax.set_xlabel("Training Steps (x1000)")
ax.set_ylabel("Average Loss")
ax.set_title("Training Loss Over Time")

model.train() # Set model to training mode

for epoch in range(epochs):
    total_loss = 0
    processed_pairs_in_epoch = 0
    for entry in train_data:
        text = entry.get('text', '')
        if not text:
            raise ValueError("No text found in entry", entry)
            continue
        
        words = text.split()
        
        if not words:
            print("No words found in text", text)
            continue

        for i, center_word_str in enumerate(words):
            if center_word_str not in word_to_ix:
                continue 

            center_word_idx = torch.tensor([word_to_ix[center_word_str]], dtype=torch.long)

            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i == j: # Skip the center word itself
                    continue
                
                context_word_str = words[j]
                if context_word_str not in word_to_ix:
                    continue # Skip if context word is not in vocab

                context_word_idx = torch.tensor([word_to_ix[context_word_str]], dtype=torch.long)

                # Forward pass
                model.zero_grad()
                output_scores = model(center_word_idx) # Shape: (1, vocab_size)
                
                loss = loss_function(output_scores, context_word_idx)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                processed_pairs_in_epoch += 1
                if processed_pairs_in_epoch % 1000 == 0: # Update plot every 1000 processed pairs
                    current_avg_loss = total_loss / processed_pairs_in_epoch
                    print(f"Epoch {epoch+1}, Step {processed_pairs_in_epoch}, Avg Loss: {current_avg_loss:.4f}, LR: {learning_rate:.6f}")
                    
                    plot_steps.append((cumulative_processed_pairs_for_plot + processed_pairs_in_epoch) / 1000) # Store cumulative steps in thousands
                    plot_losses.append(current_avg_loss)
                    
                    # Update plot data
                    line.set_xdata(plot_steps)
                    line.set_ydata(plot_losses)
                    ax.relim() # Recalculate limits
                    ax.autoscale_view(True,True,True) # Autoscale
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01) # Pause briefly to allow plot to update
    
    if processed_pairs_in_epoch > 0:
        avg_loss = total_loss / processed_pairs_in_epoch
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        cumulative_processed_pairs_for_plot += processed_pairs_in_epoch # Update cumulative count for plotting
    else:
        print(f"Epoch {epoch+1}/{epochs}, No training pairs processed.")

print("Training finished.")

# Save the final plot
plt.ioff() # Turn off interactive mode
plt.savefig("training_loss_plot.png")
print("Final loss plot saved as training_loss_plot.png")
plt.show() # Keep the plot window open

# Save the model
model_save_path = "embedding_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Save the embedding matrix
embedding_matrix = model.embeddings.weight.data.numpy()
embedding_matrix_save_path = "embedding_matrix.npy"
np.save(embedding_matrix_save_path, embedding_matrix) # Requires numpy, ensure it's imported
print(f"Embedding matrix saved to {embedding_matrix_save_path}")


