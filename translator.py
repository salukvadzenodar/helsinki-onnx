import os
from abc import ABC, abstractmethod
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer, MBart50TokenizerFast
import numpy as np

class Translator(ABC):
    @abstractmethod
    def translate(self, texts: list[str]) -> list[str]:
        pass

class HelsinkiTranslator(Translator):
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

def load_helsinki_onnx_translator(model_path: str, hf_model_name: str) -> Translator:
    tokenizer = MarianTokenizer.from_pretrained(hf_model_name)
    session = ort.InferenceSession(model_path)

    return HelsinkiTranslator(session, tokenizer)


class MBartTranslator(Translator):
    def __init__(self, model_path: str, src_lang: str = "en_XX", tgt_lang: str = "fr_XX"):
        self.model_path = model_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        self.tokenizer.src_lang = src_lang
        self.forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]

        # ONNX model
        self.model = ORTModelForSeq2SeqLM.from_pretrained(model_path)

    def translate(self, texts: list[str], max_length: int = 128) -> list[str]:
        results = []

        for text in texts:
            # Use PyTorch tensors
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Generate translation
            translated_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                forced_bos_token_id=self.forced_bos_token_id
            )

            # Decode
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            results.append(translated_text)

        return results

def load_mbrat_onnx_translator(model_path: str, src_lang: str, target_lang: str) -> Translator:
    return MBartTranslator(model_path, src_lang, target_lang)