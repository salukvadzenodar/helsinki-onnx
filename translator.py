import onnxruntime as ort
from transformers import MarianTokenizer
import numpy as np

class Translator:
    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def translate(self, texts):
        # Tokenize input
        batch = self.tokenizer(texts, return_tensors="np", padding=True)
        
        # Prepare decoder_input_ids (start token)
        decoder_input_ids = np.array([[self.session.get_inputs()[2].shape[1] if len(self.session.get_inputs()) > 2 else self.tokenizer.pad_token_id]]*len(texts))
        
        # Build ONNX inputs
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if len(self.session.get_inputs()) > 2:
            inputs["decoder_input_ids"] = decoder_input_ids
        
        # Run inference
        outputs = self.session.run(None, inputs)
        logits = outputs[0]

        # Get predicted token IDs
        pred_ids = np.argmax(logits, axis=-1)
        
        # Decode to text
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]

def load_helsinki_onnx_translator(model_path: str, hf_model_name: str):
    tokenizer = MarianTokenizer.from_pretrained(hf_model_name)
    session = ort.InferenceSession(model_path)

    return Translator(session, tokenizer)
