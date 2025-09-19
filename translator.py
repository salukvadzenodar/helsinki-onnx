import onnxruntime as ort
from transformers import MarianTokenizer
import numpy as np

class Translator:
    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def estimate_max_length(self, texts: list[str], multiplier: int = 2, min_len: int = 50, max_len: int = 200) -> int:
        text = " ".join(texts).split()
        num_words = len(text)
        est_len = num_words * multiplier

        return max(min_len, min(est_len, max_len))

    def translate(self, texts: list[str]):
        batch = self.tokenizer(texts, return_tensors="np", padding=True)
        input_ids = batch["input_ids"].astype(np.int64)
        attention_mask = batch["attention_mask"].astype(np.int64)

        # Initialize decoder input with start token
        decoder_input_ids = np.full(
            (len(texts), 1), self.tokenizer.pad_token_id, dtype=np.int64
        )

        # Keep track of generated tokens
        output_ids = decoder_input_ids.copy()
        max_length = self.estimate_max_length(texts)

        for _ in range(max_length):
            # Prepare ONNX inputs
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": output_ids,
            }

            # Run decoder
            logits = self.session.run(None, inputs)[0]  # [batch_size, seq_len, vocab_size]

            # Get last token prediction
            next_token_id = np.argmax(logits[:, -1, :], axis=-1).reshape(-1, 1)

            # Append predicted token
            output_ids = np.concatenate([output_ids, next_token_id], axis=1)

            # Stop if all sequences generated eos token
            if np.all(next_token_id == self.tokenizer.eos_token_id):
                break

        # Remove the initial pad token
        output_ids = output_ids[:, 1:]

        # Decode to text
        translations = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]

        return translations

def load_helsinki_onnx_translator(model_path: str, hf_model_name: str):
    tokenizer = MarianTokenizer.from_pretrained(hf_model_name)
    session = ort.InferenceSession(model_path)

    return Translator(session, tokenizer)
