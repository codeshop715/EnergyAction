import torch
from torch import nn
import transformers
import os


def _get_clip_cache_path():
    """Get the CLIP model cache path, with fallback to HuggingFace download."""
    local_cache = os.path.join(os.path.dirname(__file__), "../../../pretrain_unimanual/openai/clip-vit-base-patch32")
    if os.path.exists(local_cache):
        return local_cache
    return None


class ClipTokenizer:

    def __init__(self):
        super().__init__()
        
        master_cache = _get_clip_cache_path()
        
        if master_cache:
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(master_cache)
        else:
            print("Downloading CLIP tokenizer from HuggingFace...")
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="~/.cache/huggingface"
            )
        print("CLIP tokenizer loaded successfully.")

    @torch.inference_mode()
    def __call__(self, instructions):
        # Handle empty or None instructions
        if instructions is None or (isinstance(instructions, list) and len(instructions) == 0):
            instructions = [""]
        elif isinstance(instructions, str) and instructions == "":
            instructions = [""]
        elif isinstance(instructions, list) and any(inst is None or inst == "" for inst in instructions):
            # Replace None or empty strings with empty string
            instructions = [inst if inst is not None and inst != "" else "" for inst in instructions]
        
        tokens = self.tokenizer(
            instructions,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]
        
        return tokens


class ClipTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        master_cache = _get_clip_cache_path()
        
        if master_cache:
            self.model = transformers.CLIPTextModel.from_pretrained(master_cache)
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(master_cache)
        else:
            print("Downloading CLIP text model from HuggingFace...")
            self.model = transformers.CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="~/.cache/huggingface"
            )
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="~/.cache/huggingface"
            )
        print("CLIP text encoder loaded successfully.")

    def forward(self, text_or_tokens):
        """
        Forward pass that accepts either text strings or pre-tokenized tokens.
        
        Args:
            text_or_tokens: Either a list of strings or a tensor of token IDs
        """
        # Check if input is text (list of strings) or tokens (tensor)
        if isinstance(text_or_tokens, (list, str)):
            # Tokenize text input
            if isinstance(text_or_tokens, str):
                text_or_tokens = [text_or_tokens]
            tokens = self.tokenizer(
                text_or_tokens,
                padding="longest",
                return_tensors="pt"
            )["input_ids"]
            # Move tokens to same device as model
            tokens = tokens.to(self.model.device)
        else:
            # Already tokenized
            tokens = text_or_tokens
        
        return self.model(input_ids=tokens).last_hidden_state
