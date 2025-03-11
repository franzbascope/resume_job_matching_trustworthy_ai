import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class ContrastiveModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=768):
        super(ContrastiveModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(embedding_dim, 512)
        
    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling: average all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply mean pooling
        embeddings = self.mean_pooling(outputs, attention_mask)
        
        # Project to lower dimension and normalize
        embeddings = self.proj(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    # for inference 
    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)
