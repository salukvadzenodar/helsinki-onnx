from downloader import list_helsinki_models, download_helsinki_model
from converter import convert_helsinki_to_tf


def main():
    language_code = input('filter language code or skip: ').strip()
    availabel_models = list_helsinki_models(language_code if language_code != "skip" else "")
    
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

    dir_name = input("enter name to download: ")
    if dir_name == "":
        dir_name = language[0]
    pt_name = download_helsinki_model(language[1], dir_name)
    print("pt path =", pt_name)

    convert = input("do you want to convert model? Options: tf, tflite, no: ").strip()
    if convert in ("tf", "tflite"):
        tf_path = convert_helsinki_to_tf(pt_name, convert == "tflite")
        print("tf path =", tf_path)

if __name__ == "__main__":
    main()