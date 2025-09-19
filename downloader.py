import os
from huggingface_hub import HfApi, snapshot_download


def list_helsinki_models(code: str = "", big = True) -> list[tuple[str, str]]:
    api = HfApi()

    # List all models with "Helsinki-NLP/opus-mt" in their repo_id
    models = api.list_models(search=f"Helsinki-NLP/opus-mt{"-tc-big" if big else ""}")
    language_pairs = []

    for model in models:
        parts = model.modelId.split(f"opus-mt-{"tc-big-" if big else ""}")

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


def download_mbart50_mtm_model(llm_dir = "llm_models"):
    dir_path = os.path.join(llm_dir, "mbart50_mtm")
    os.makedirs(llm_dir, exist_ok=True)

    if os.path.exists(dir_path):
        return os.path.abspath(dir_path)
    
    snapshot_download(repo_id="facebook/mbart-large-50-many-to-many-mmt", local_dir=dir_path)
    return os.path.abspath(dir_path)

def download_mgpt_georgian_model(llm_dir = "llm_models"):
    dir_path = os.path.join(llm_dir, "mgpt_georgian")
    os.makedirs(llm_dir, exist_ok=True)

    if os.path.exists(dir_path):
        return os.path.abspath(dir_path)
    
    snapshot_download(repo_id="ai-forever/mGPT-1.3B-georgian", local_dir=dir_path)
    return os.path.abspath(dir_path)