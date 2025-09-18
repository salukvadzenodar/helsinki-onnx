import os
from transformers import TFAutoModelForSeq2SeqLM
from optimum.exporters.tflite import TFLiteExporter


def convert_helsinki_to_tf(pt_model_path: str, light = False, sequence_length = 128) -> str:
    # Base name for saving
    base_dir = os.path.dirname(pt_model_path)
    base_name = os.path.basename(pt_model_path.rstrip("/\\"))
    
    if not light:
        # Convert to TensorFlow
        tf_model = TFAutoModelForSeq2SeqLM.from_pretrained(pt_model_path, from_pt=True)

        # Save as TensorFlow SavedModel folder
        tf_model_path = os.path.join(base_dir, base_name + "_tf")
        os.makedirs(tf_model_path, exist_ok=True)
        tf_model.save_pretrained(tf_model_path)
        return tf_model_path

    # Convert to TFLite
    tflite_path = os.path.join(base_dir, base_name + ".tflite")
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

    TFLiteExporter.export(
        model_name_or_path=pt_model_path,
        output=tflite_path,
        task="translation",
        sequence_length=128
    )

    return tflite_path
    


# import os
# from optimum.exporters.tflite import TFLiteExporter
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from optimum.exporters.tasks import TasksManager

# def convert_helsinki_to_tflite(model_name_or_path: str, output_dir: str, sequence_length: int = 128) -> str:

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Define the full path for the output TFLite file
#     tflite_path = os.path.join(output_dir, "model.tflite")

#     # Initialize the TFLiteExporter
#     exporter = TFLiteExporter.from_pretrained(
#         model_name_or_path,
#         task="text2text-generation"
#     )

#     # Perform the export directly to TFLite
#     exporter.export(
#         output_path=tflite_path,
#         sequence_length=sequence_length,
#         # You can add quantization here if desired.
#         # quantization_config=TFLiteQuantizationConfig(mode="int8-dynamic")
#     )

#     print(f"✅ Conversion successful! TFLite model saved at: {tflite_path}")
#     return tflite_path
