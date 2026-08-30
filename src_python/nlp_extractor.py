import torch
from transformers import AutoTokenizer, AutoModel

class InstitutionalNLPEngine:
    def __init__(self):
        print("[NLP PIPELINE] Initializing FinBERT model for sentiment extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def extract_earnings_shock_embedding(self, transcript_text):
        """
        Parses raw text (e.g., earnings call transcript) and extracts a 
        768-dimensional [CLS] tensor representing institutional sentiment.
        """
        inputs = self.tokenizer(
            transcript_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract the [CLS] token representation (hidden state of the first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

if __name__ == "__main__":
    engine = InstitutionalNLPEngine()
    sample_transcript = "The company experienced severe supply chain compression, resulting in lower forward guidance and an unexpected margin squeeze for the upcoming fiscal quarters."
    vector = engine.extract_earnings_shock_embedding(sample_transcript)
    print(f"Extracted NLP Shock Vector Shape: {vector.shape}")