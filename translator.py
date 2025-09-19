import os
from abc import ABC, abstractmethod
import onnxruntime as ort
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
    def __init__(self, model_path: str, src_lang: str = "en_XX", target_lang: str = "en_XX", top_k: int = 5):
        self.model_path = model_path
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.top_k = top_k

        # Load tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        self.tokenizer.src_lang = self.src_lang

        # Forced BOS token = target language
        self.forced_bos_id = self.tokenizer.lang_code_to_id[self.target_lang]

        # Load ONNX encoder + decoder (no past)
        self.encoder_sess = ort.InferenceSession(
            os.path.join(model_path, "encoder_model.onnx"),
            providers=["CPUExecutionProvider"]
        )
        self.decoder_sess = ort.InferenceSession(
            os.path.join(model_path, "decoder_model.onnx"),
            providers=["CPUExecutionProvider"]
        )

    def _sample_next_token(self, logits: np.ndarray, generated_ids: list[int]) -> int:
        """
        Sample next token from top-k logits with repetition penalty
        """
        logits_copy = logits.copy()
        for token_id in set(generated_ids):
            logits_copy[token_id] /= 1.2  # repetition penalty

        top_k = min(self.top_k, logits_copy.size)
        top_ids = np.argpartition(-logits_copy, top_k)[:top_k]
        top_probs = logits_copy[top_ids]
        top_probs = np.exp(top_probs - np.max(top_probs))
        top_probs /= top_probs.sum()
        return int(np.random.choice(top_ids, p=top_probs))

    def translate(self, texts: list[str]) -> list[str]:
        results = []

        for text in texts:
            # Encode input
            inputs = self.tokenizer(text, return_tensors="np")
            encoder_hidden_states = self.encoder_sess.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
            )[0]

            # Start decoder with forced BOS token (target language)
            decoder_input_ids = np.array([[self.forced_bos_id]], dtype=np.int64)
            generated_ids = []

            for _ in range(128):  # max length
                feed_dict = {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": inputs["attention_mask"]
                }

                outputs = self.decoder_sess.run(None, feed_dict)
                logits = outputs[0]

                # Sample next token
                next_token_id = self._sample_next_token(logits[0, -1, :], generated_ids)
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)
                decoder_input_ids = np.hstack([decoder_input_ids, [[next_token_id]]])

            results.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True))

        return results
    

def load_mbrat_onnx_translator(model_path: str, src_lang: str, target_lang: str) -> Translator:
    return MBartTranslator(model_path, src_lang, target_lang)