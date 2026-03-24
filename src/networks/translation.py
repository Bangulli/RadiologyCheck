from transformers import MarianMTModel, MarianTokenizer
import re

class Translator():
    def __init__(self, model_name = "models/tran", device = 'cuda:1'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name, device_map=device)
        self.device = device
        self.max_tokens = 128
        
    def _chunkify(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            token_len = len(self.tokenizer(sentence).input_ids)
            if current_length + token_len > self.max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = token_len
            else:
                current_chunk.append(sentence)
                current_length += token_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    
    def _translate_chunk(self, chunk):
        tok = self.tokenizer(chunk, return_tensors="pt").input_ids
        output = self.model.generate(tok.to(self.device))[0]
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def __call__(self, text): 
        chunks = self._chunkify(text)
        return " ".join(self._translate_chunk(c) for c in chunks)