from sentence_transformers import SentenceTransformer
import numpy as np

class TextVectorizer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text):
        return self.model.encode(text).tolist()