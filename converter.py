import os
from shutil import move
import torch
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.onnxruntime import ORTModelForCausalLM
from onnxruntime.quantization import quantize_dynamic, QuantType


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
    onnx__dir = os.path.join(base_dir, f"{base_name}_onnx")

    onnx_model = AutoModelForSeq2SeqLM.from_pretrained(pt_model_path, export=True)
    tokenizer = AutoTokenizer.from_pretrained(pt_model_path)

    # Save the ONNX model and tokenizer to the new directory
    onnx_model.save_pretrained(onnx__dir)
    tokenizer.save_pretrained(onnx__dir)

    if quantize:
        for f in ["encoder_model.onnx", "decoder_with_past_model.onnx"]:
            onnx_name = os.path.join(onnx__dir, f)
            quantized_name = os.path.join(onnx__dir, f"{f.replace(".onnx", "")}-qint8.onnx")
            quantize_dynamic(onnx_name, quantized_name, weight_type=QuantType.QInt8)
    
    os.remove(os.path.join(onnx__dir, "decoder_model.onnx"))
    return onnx__dir

def convert_mgpt_to_onnx(pt_model_path: str, quantize=True) -> str:
    base_dir = os.path.dirname(pt_model_path)
    base_name = os.path.basename(pt_model_path.rstrip("/\\"))
    onnx_file_name = os.path.join(base_dir, f"{base_name}.onnx")

    # Load PyTorch GPT model
    # pt_model = GPT2LMHeadModel.from_pretrained(pt_model_path)

    # Export to ONNX folder
    ORTModelForCausalLM.from_pretrained(pt_model_path, export=True, cache_dir=base_dir)

    # Rename
    generated_file = os.path.join(base_dir, "model.onnx")
    move(generated_file, onnx_file_name)

    if quantize:
        quantized_file_name = os.path.join(base_dir, f"{base_name}-qint8.onnx")
        quantize_dynamic(onnx_file_name,quantized_file_name,weight_type=QuantType.QInt8)
        return quantized_file_name
    else:
        return onnx_file_name