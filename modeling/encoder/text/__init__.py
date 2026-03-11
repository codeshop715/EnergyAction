from .clip import ClipTextEncoder, ClipTokenizer


def fetch_text_encoders(model_name):
    """Return encoder class and latent dimension."""
    if model_name == 'clip':
        return ClipTextEncoder(), 512
    return None, None


def fetch_tokenizers(model_name):
    """Return tokenizer class."""
    if model_name == 'clip':
        return ClipTokenizer()
    return None
