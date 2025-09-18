import os
from transformers import MarianMTModel, TFAutoModelForSeq2SeqLM
import tensorflow as tf


def convert_helsinki_to_tf(pt_model_path: str, light = False) -> str:
    # Base name for saving
    base_dir = os.path.dirname(pt_model_path)
    base_name = os.path.basename(pt_model_path.rstrip("/\\"))

    # Convert to TensorFlow
    tf_model = TFAutoModelForSeq2SeqLM.from_pretrained(pt_model_path, from_pt=True)
    
    if not light:
        # Save as TensorFlow SavedModel folder
        tf_model_path = os.path.join(base_dir, base_name + "_tf")
        os.makedirs(tf_model_path, exist_ok=True)
        tf_model.save_pretrained(tf_model_path)
        return tf_model_path

    # Convert to TFLite
    tflite_path = os.path.join(base_dir, base_name + ".tflite")
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]  # reduce memory usage
    tflite_model = converter.convert()

    # Save the TFLite model as a single file
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    return tflite_path

