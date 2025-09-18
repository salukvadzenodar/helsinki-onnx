import os
from huggingface_hub import HfApi, snapshot_download


def list_helsinki_models(code: str = "") -> list[tuple[str, str]]:
    api = HfApi()

    # List all models with "Helsinki-NLP/opus-mt" in their repo_id
    models = api.list_models(search="Helsinki-NLP/opus-mt")
    language_pairs = []

    for model in models:
        # model.modelId is like 'Helsinki-NLP/opus-mt-en-ka'
        parts = model.modelId.split("opus-mt-")

        if len(parts) == 2:
            lang_pair = parts[1]
            language_pairs.append((lang_pair, model.modelId))

    if len(code) > 0:
        language_pairs = list(filter(lambda x: code in x[0], language_pairs))

    return language_pairs

def download_helsinki_model(repo: str, model_dir: str, llm_dir = "llm_models"):
    dir_path = os.path.join(llm_dir, model_dir)
    os.makedirs(llm_dir, exist_ok=True)

    if os.path.exists(dir_path):
        return os.path.abspath(dir_path)
    
    snapshot_download(repo_id=repo, local_dir=dir_path)
    return os.path.abspath(dir_path)