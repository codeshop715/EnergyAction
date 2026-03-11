from .clip import load_clip


def fetch_visual_encoders(model_name):
    if model_name == "clip":
        return load_clip()
    return None, None
