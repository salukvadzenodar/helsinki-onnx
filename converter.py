import os, glob, shutil
import torch
import onnx
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM
from onnxruntime.quantization import quantize_dynamic, QuantType

def combine_onnx_external_data(input_onnx_path, output_onnx_path):
    model = onnx.load(input_onnx_path, load_external_data=True)
    # Converts all external data to internal (embed weights)
    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,          # store all tensors in a single internal file
        size_threshold=0,                      # embed all tensors regardless of size
    )
    onnx.save(model, output_onnx_path)


def convert_helsinki_to_onnx(pt_model_path: str, quantize = True) -> str:
    # Base name for saving
    base_dir = os.path.dirname(pt_model_path)
    base_name = os.path.basename(pt_model_path.rstrip("/\\"))
    onnx_file_name = os.path.join(base_dir, f"{base_name}.onnx")

    pt_model = MarianMTModel.from_pretrained(pt_model_path)
    pt_tokenizer = MarianTokenizer.from_pretrained(pt_model_path)

    # Example input
    text = ["Hello world"]
    batch = pt_tokenizer(text, return_tensors="pt")

    # Create decoder_input_ids
    decoder_input_ids = torch.tensor([[pt_model.config.decoder_start_token_id]] * batch["input_ids"].shape[0])

    # Before exporting, set use_cache=False
    pt_model.config.use_cache = False

    # Export to ONNX
    torch.onnx.export(
        pt_model,
        (batch["input_ids"], batch["attention_mask"], decoder_input_ids),  # model inputs
        onnx_file_name,                                # output file
        input_names=["input_ids", "attention_mask", "decoder_input_ids"],  
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=14
    )

    if quantize:
        quantized_file_name = os.path.join(base_dir, f"{base_name}-qint8.onnx")
        quantize_dynamic(onnx_file_name,quantized_file_name,weight_type=QuantType.QInt8)
        return quantized_file_name
    else:
        return onnx_file_name
        

def convert_mbart_to_onnx(pt_model_path: str, quantize = True) -> str:
    # Base name for saving
    base_dir = os.path.dirname(pt_model_path)
    base_name = os.path.basename(pt_model_path.rstrip("/\\"))
    onnx_dir = os.path.join(base_dir, f"{base_name}_onnx")

    if(not quantize):
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(pt_model_path, export=True)
        tokenizer = AutoTokenizer.from_pretrained(pt_model_path)

        # Save the ONNX model and tokenizer to the new directory
        onnx_model.save_pretrained(onnx_dir)
        tokenizer.save_pretrained(onnx_dir)
        return onnx_dir
    
    onnx_dir_quantized = os.path.join(base_dir, f"{base_name}_onnx_qint8")
    files_to_quantize = ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]

    if os.path.exists(onnx_dir_quantized):
        shutil.rmtree(onnx_dir_quantized)
    os.makedirs(onnx_dir_quantized, exist_ok=True)
    
    all_files = os.listdir(onnx_dir)
    for file_name in all_files:
        print(file_name)
        if file_name.endswith(".onnx_data"):
            continue
    
        source_path = os.path.join(onnx_dir, file_name)
        destination_path = os.path.join(onnx_dir_quantized, file_name)

        if file_name not in files_to_quantize:
            shutil.copy(source_path, destination_path)
            continue

        file_combined = False
        if os.path.exists(source_path.replace(".onnx", ".onnx_data")): # check if data file exists
            combined_path = source_path.replace(".onnx", "_combined.onnx")
            print(combined_path)
            combine_onnx_external_data(source_path, combined_path)
            file_combined = True
        else:
            combined_path = source_path

        print("start quantization")
        quantize_dynamic(combined_path, destination_path, weight_type=QuantType.QInt8)
        if file_combined:
            os.remove(combined_path)

    for f in glob.glob(os.path.join(onnx_dir, "*.data")):
        if os.path.isfile(f):
            os.remove(f)

    return onnx_dir_quantized


def convert_mgpt_to_onnx(pt_model_path: str, quantize=True) -> str:
   # Base name for saving
    base_dir = os.path.dirname(pt_model_path)
    base_name = os.path.basename(pt_model_path.rstrip("/\\"))
    onnx_dir = os.path.join(base_dir, f"{base_name}_onnx")

    if(not quantize):
        onnx_model = ORTModelForCausalLM.from_pretrained(pt_model_path, export=True)
        tokenizer = AutoTokenizer.from_pretrained(pt_model_path)

        # Save the ONNX model and tokenizer to the new directory
        onnx_model.save_pretrained(onnx_dir)
        tokenizer.save_pretrained(onnx_dir)
        return onnx_dir

    onnx_dir_quantized = os.path.join(base_dir, f"{base_name}_onnx_qint8")
    files_to_quantize = ["model.onnx"]

    if os.path.exists(onnx_dir_quantized):
        shutil.rmtree(onnx_dir_quantized)
    os.makedirs(onnx_dir_quantized, exist_ok=True)
    
    all_files = os.listdir(onnx_dir)
    for file_name in all_files:
        print(file_name)
        if file_name.endswith(".onnx_data"):
            continue
    
        source_path = os.path.join(onnx_dir, file_name)
        destination_path = os.path.join(onnx_dir_quantized, file_name)

        if file_name not in files_to_quantize:
            shutil.copy(source_path, destination_path)
            continue

        combined_path = source_path.replace(".onnx", "_combined.onnx")
        print(combined_path)
        combine_onnx_external_data(source_path, combined_path)
        print("start quantization")
        quantize_dynamic(combined_path, destination_path, weight_type=QuantType.QInt8)
        os.remove(combined_path)

    for f in glob.glob(os.path.join(onnx_dir, "*.data")):
        if os.path.isfile(f):
            os.remove(f)

    return onnx_dir_quantized