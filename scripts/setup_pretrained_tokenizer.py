"""
Setup a pretrained tokenizer for nanochat when training tokenizer is not possible.
Uses GPT-2 tokenizer (vocab_size=50257) or cl100k_base (vocab_size=100276).
"""
import os
import pickle
import torch
import tiktoken
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from nanochat.common import get_base_dir

# Choose tokenizer: 'gpt2' or 'cl100k_base'
TOKENIZER_NAME = 'gpt2'

def setup_pretrained_tokenizer(tokenizer_name='gpt2'):
    """Create tokenizer files for nanochat using a pretrained tiktoken tokenizer."""
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Get pretrained encoding
    enc = tiktoken.get_encoding(tokenizer_name)
    
    # Add nanochat special tokens
    # We need to extend the tokenizer with our special tokens
    vocab_size = enc.n_vocab
    special_tokens = {name: vocab_size + i for i, name in enumerate(SPECIAL_TOKENS)}
    
    # Create extended encoding
    extended_enc = tiktoken.Encoding(
        name=f"nanochat_{tokenizer_name}",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={**enc._special_tokens, **special_tokens},
    )
    
    # Save as pickle (nanochat expects this format)
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(extended_enc, f)
    
    print(f"Saved tokenizer to {pickle_path}")
    print(f"Vocab size: {extended_enc.n_vocab}")
    
    # Create token_bytes.pt for bits per byte calculation
    vocab_size_full = extended_enc.n_vocab
    special_set = set(SPECIAL_TOKENS)
    token_bytes = []
    
    for token_id in range(vocab_size_full):
        try:
            token_str = extended_enc.decode([token_id])
            if token_str in special_set or token_id >= vocab_size:
                token_bytes.append(0)  # special tokens not counted
            else:
                token_bytes.append(len(token_str.encode("utf-8")))
        except:
            token_bytes.append(0)
    
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    
    print(f"Saved token_bytes to {token_bytes_path}")
    
    # Verify
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    test_text = "Hello world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Test encode/decode: '{test_text}' -> {len(encoded)} tokens -> '{decoded}'")
    
    return tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='gpt2', help='tiktoken tokenizer name')
    args = parser.parse_args()
    setup_pretrained_tokenizer(args.name)