import os
from models import ModelTypes
from downloader import list_helsinki_models, download_helsinki_model, download_mbart50_mtm_model, download_mgpt_georgian_model
from converter import convert_helsinki_to_onnx, convert_mbart_to_onnx, convert_mgpt_to_onnx
from translator import load_helsinki_onnx_translator, load_mbrat_onnx_translator

def creation_progress(model_type: ModelTypes):
    if model_type == ModelTypes.HELSINKI:
        language_input = input('filter language code, type small for small models or skip: ').strip().split()

        if language_input:
            language_code = language_input[0]
            small = True if len(language_input) > 1 and language_input[1] == "small" else False
        else:
            language_code = ""
            small = False
        availabel_models = list_helsinki_models(language_code if language_code != "skip" else "", not small)
    
        if language_code != "skip":
            print("Available MarianMT language pairs:")
            for pair in availabel_models:
                print(pair)

            print(f"\nTotal models: {len(availabel_models)}")
        if len(availabel_models) == 0:
            return

        language_code = input('enter language code to download: ').strip()
        language = next(filter(lambda x: language_code in x[0], availabel_models), None)
        if language is None:
            print("language not found!")
            return
        print(F"chosen language: {language[0]}, {language[1]}")

        pt_name = download_helsinki_model(language[1], language[0])
    elif model_type == ModelTypes.MBART:
        pt_name = download_mbart50_mtm_model()
    elif model_type == ModelTypes.MGPT_GEORGIAN:
        pt_name = download_mgpt_georgian_model()

    print("pt path =", pt_name)
    convert = input("do you want to convert model? y, n, quantize: ").strip().lower()
    quantize = True if convert == "quantize" else False
    if convert == "quantize":
        convert = "y"

    if convert == "y":
        if model_type == ModelTypes.HELSINKI:
            onnx_path = convert_helsinki_to_onnx(pt_name, quantize)
        elif model_type == ModelTypes.MBART:
            onnx_path = convert_mbart_to_onnx(pt_name, quantize)
        elif model_type == ModelTypes.MGPT_GEORGIAN:
            onnx_path = convert_mgpt_to_onnx(pt_name, quantize)

    print("onnx path =", onnx_path)

def usage_progress(model_type: ModelTypes):
    small = False
    if model_type != ModelTypes.HELSINKI:
        small = input("Do you want to run quantized model? y, n: ").strip() == "y"

    if model_type == ModelTypes.HELSINKI:
        all_items = os.listdir("llm_models")
        files = [f for f in all_items if os.path.isfile(os.path.join("llm_models", f))]
        print(files)

        file_name = input("Enter file name to test or quit: ").strip().lower()
        if file_name in ("quit", "exit") or file_name not in files:
            return
        
        language_code = file_name.replace(".onnx", "").replace("-qint8", "")
        availabel_models = list_helsinki_models(language_code, False)
        if len(availabel_models) != 1:
            print("language not available")
            return
        translator = load_helsinki_onnx_translator(os.path.join("llm_models", file_name), availabel_models[0][1])
    elif model_type == ModelTypes.MBART:
        src_lang = input("Choose source language! English: en_XX, German: de_DE, French: fr_XX, Russian: ru_RU, Spanish: es_XX, Georgian: ka_GE\n").strip()
        if len(src_lang) == 0:
            src_lang = "en_XX"

        target_lang = input("Choose target language! English: en_XX, German: de_DE, French: fr_XX, Russian: ru_RU, Spanish: es_XX, Georgian: ka_GE\n").strip()
        if len(target_lang) == 0:
            target_lang = "en_XX"

        translator = load_mbrat_onnx_translator(os.path.join("llm_models", "mbart50_mtm_onnx_qint8" if small else "mbart50_mtm_onnx"), src_lang, target_lang)

    print("AI is ready to assist:\n\n")

    while True:
        text = input("Enter text to translate or exit: ").strip()
        if text.lower() == "exit":
            break

        translation = translator.translate([text])
        print(f"Translated: {translation[0]}")


def main():
    choose_model_input = input("choose the model you want to use: 1 = Helsinki, 2 = mBART-50 many to many, 3 = mGPT Georgian: ").strip()
    if choose_model_input == "1":
        chosen_model = ModelTypes.HELSINKI
    elif choose_model_input == "2":
        chosen_model = ModelTypes.MBART
    elif choose_model_input == "3":
        chosen_model = ModelTypes.MGPT_GEORGIAN
    else:
        return

    progress_path = input("do you want to download model? y, n: ").strip()
    if progress_path == "y":
        creation_progress(chosen_model)
    else:
        usage_progress(chosen_model)


if __name__ == "__main__":
    main()