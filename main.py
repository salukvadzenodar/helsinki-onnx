import os
from downloader import list_helsinki_models, download_helsinki_model
from converter import convert_helsinki_to_onnx
from translator import load_helsinki_onnx_translator

def creation_progress():
    language_input = input('filter language code, type small for small models or skip: ').strip().split() 
    language_code, *rest = language_input
    small = True if rest and rest[0] == "small" else False
    availabel_models = list_helsinki_models(language_code if language_code != "skip" else "", not small)
    
    if language_code != "skip":
         print("Available MarianMT language pairs:")
         for pair in availabel_models:
            print(pair)

         print(f"\nTotal models: {len(availabel_models)}")

    language_code = input('enter language code to download: ').strip()
    language = next(filter(lambda x: language_code in x[0], availabel_models), None)
    if language is None:
        print("language not found!")
        return
    print(F"chosen language: {language[0]}, {language[1]}")

    pt_name = download_helsinki_model(language[1], language[0])
    print("pt path =", pt_name)

    convert = input("do you want to convert model? y, n: ").strip()
    if convert == "y":
        onnx_path = convert_helsinki_to_onnx(pt_name)
        print("onnx path =", onnx_path)

def usage_progress():
    all_items = os.listdir("llm_models")
    files = [f for f in all_items if os.path.isfile(os.path.join("llm_models", f))]
    print(files)

    file_name = input("Enter file name to test or quit: ").strip().lower()
    if file_name in ("quit", "exit") or file_name not in files:
        return
    
    language_code = file_name.replace(".onnx", "").replace("-qint8", "")
    availabel_models = list_helsinki_models(language_code)
    if len(availabel_models) != 1:
        print("language not available")
        return
    
    translator = load_helsinki_onnx_translator(os.path.join("llm_models", file_name), availabel_models[0][1])
    print("AI is ready to assist:\n\n")

    while True:
        text = input("Enter text to translate or exit: ").strip()
        if text.lower() == "exit":
            break

        translation = translator.translate([text])
        print(f"Translated: {translation[0]}")


def main():
    progress_path = input("do you want to download model? y, n: ").strip()
    if progress_path == "y":
        creation_progress()
    else:
        usage_progress()


if __name__ == "__main__":
    main()