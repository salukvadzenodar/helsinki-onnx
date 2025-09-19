import os
import torch
from transformers import MarianMTModel, MarianTokenizer
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
