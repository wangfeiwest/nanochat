#!/usr/bin/env python3
"""Test script to verify AdamW optimizer works on ROCm (no NaN loss)"""
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'
os.environ['WANDB_MODE'] = 'disabled'

import torch
import torch.nn as nn

# Simple test model
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size=32768, n_embd=256, n_head=4, n_layer=4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(n_embd, n_head, n_embd*4, batch_first=True)
            for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
    def forward(self, x, targets=None):
        h = self.wte(x)
        for block in self.blocks:
            h = block(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        return logits, loss

print("Creating model...")
device = 'cuda'
model = SimpleGPT().to(device)
model = model.to(torch.float32)

print("Creating optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Testing training loop...")
batch_size = 4
seq_len = 128

for step in range(10):
    x = torch.randint(0, 32768, (batch_size, seq_len), device=device)
    y = torch.randint(0, 32768, (batch_size, seq_len), device=device)
    
    optimizer.zero_grad()
    logits, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
    
    if torch.isnan(loss):
        print(f"FAILED: Step {step}: NaN loss detected!")
        break
    else:
        print(f"Step {step}: loss = {loss.item():.4f}")
else:
    print("\nSUCCESS: No NaN loss with AdamW optimizer on ROCm!")