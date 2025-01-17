import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import tiktoken
from transformer import Config, DecoderOnlyTransformer

# Data Loader
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        # Load tokens from the file
        with open('/kaggle/input/input-txt/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        # Shuffle tokens while preserving sequence structure
        chunk_size = len(self.tokens) // B
        chunks = torch.split(self.tokens, chunk_size)
        shuffled_chunks = [chunks[i] for i in torch.randperm(len(chunks))]
        self.tokens = torch.cat(shuffled_chunks)
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + B * T + 1 > len(self.tokens):
            # Reshuffle tokens
            chunk_size = len(self.tokens) // B
            chunks = torch.split(self.tokens, chunk_size)
            shuffled_chunks = [chunks[i] for i in torch.randperm(len(chunks))]
            self.tokens = torch.cat(shuffled_chunks)
            self.current_position = 0
        
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        return x, y


# Training Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")
torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed(1337)

model = DecoderOnlyTransformer(Config())
model.to(device)

batches, no_of_tokens = 16, 128
train_loader = DataLoaderLite(B=batches, T=no_of_tokens)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
scaler = torch.amp.GradScaler()

# Training Loop
steps_per_epoch = len(train_loader.tokens) // (batches * no_of_tokens)
for epoch in range(100):
    loss_list = []
    start_time = time.time()
    for step in range(steps_per_epoch):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # Mixed precision context
        with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backpropagation with scaled gradients
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        loss_list.append(loss.item())
    
    scheduler.step()
    epoch_loss = sum(loss_list) / len(loss_list)
    perplexity = math.exp(epoch_loss) if epoch_loss < 20 else float('inf')
    print(f"Epoch {epoch + 1}/{100}, Loss: {epoch_loss:.4f}, Perplexity: {perplexity:.2f}, Time: {time.time() - start_time:.2f}s")