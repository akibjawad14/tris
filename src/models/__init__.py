import json


def load_json(file_path):
    with open(file_path, encoding='utf-8') as file:
        return json.load(file)


def create_model(config_path):
    """Create an LLM client from a local JSON configuration file.

    Imports are intentionally lazy so collaborators only need dependencies for
    the provider they actually use.
    """
    config = load_json(config_path)
    provider = config["model_info"]["provider"].lower()

    if provider == "gpt":
        from .GPT import GPT
        return GPT(config)
    if provider == "palm2":
        from .PaLM2 import PaLM2
        return PaLM2(config)
    if provider == "vicuna":
        from .Vicuna import Vicuna
        return Vicuna(config)
    if provider == "llama":
        from .Llama import Llama
        return Llama(config)
    raise ValueError(f"Unknown provider: {provider}")
