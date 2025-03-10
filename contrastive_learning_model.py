#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup


# In[2]:


class ContrastiveModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', embedding_dim=768):
        super(ContrastiveModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(embedding_dim, 512)

    def mean_pooling(self, model_output, attention_mask):
        # Average token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        # Forward pass

        # Get contextual representations
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Combine encoder outputs into one sentence level embedding
        embeddings = self.mean_pooling(outputs, attention_mask)

        # map embeddings from bert native to 512
        embeddings = self.proj(embeddings)
        # L2 normalization
        embedding = F.normalize(embeddings, p=2, dim=1)

        return embeddings


# In[3]:


model = ContrastiveModel()
test_id = torch.randint(0, 1000, (2,128))
test_mask = torch.ones_like(test_id)
embeddings = model(test_id, test_mask)


# In[4]:


embeddings.shape


# In[5]:


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, loss_type='contrastive'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_type = loss_type

    def forward(self, anchor, positive, negative=None, labels=None):
        if self.loss_type == 'contrastive':
            distances = torch.norm(anchor - positive, dim=1)
            
            # For positive labels minimize, for negative ensure distance is at least margin
            losses = labels * distances + (1 -labels) * F.relu(self.margin - distances)
            return losses.mean()
            
        elif self.loss_type == 'cosine':
            cos_sim = F.cosine_similarity(anchor, positive)
            # maximize similarity
            return -cos_sim.mean()
        


# In[6]:


loss_fn = ContrastiveLoss(margin=0.5, loss_type='contrastive')
anchor = torch.randn(4, 512)
positive = torch.randn(4, 512)
labels = torch.tensor([1, 1, 0, 0], dtype=torch.float)
loss = loss_fn(F.normalize(anchor, dim=1), F.normalize(positive, dim=1), labels=labels)
loss.item()


# In[7]:


class ResumeJobDataset(Dataset):
    def __init__(self, pairs_df, resume_df, job_df, tokenizer, max_length=256):
        self.pairs_df = pairs_df
        self.resume_df = resume_df
        self.job_df = job_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        pair = self.pairs_df.iloc[idx]
        resume_id = pair['resume_id']
        job_id = pair['job_id']
        label = pair['label']
        
        # Get resume text
        resume_text = self.resume_df.loc[self.resume_df['ID'] == resume_id, 'Resume_str'].values[0]
        
        # Get job text
        job_title = self.job_df.loc[self.job_df['Job Id'] == job_id, 'Job Title'].values[0]
        job_desc = self.job_df.loc[self.job_df['Job Id'] == job_id, 'Job Description'].values[0]
        job_text = f"{job_title}. {job_desc}"
        
        # Tokenize
        resume_encoding = self.tokenizer.encode_plus(
            resume_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        job_encoding = self.tokenizer.encode_plus(
            job_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'resume_input_ids': resume_encoding['input_ids'].squeeze(),
            'resume_attention_mask': resume_encoding['attention_mask'].squeeze(),
            'job_input_ids': job_encoding['input_ids'].squeeze(),
            'job_attention_mask': job_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }


# In[8]:


def load_data(pairs_path, resume_path, job_path):
    pairs_df = pd.read_csv(pairs_path)
    resume_df = pd.read_csv(resume_path)
    job_df = pd.read_csv(job_path)
    return pairs_df, resume_df, job_df


# In[9]:


pairs_df, resume_df, job_df = load_data(
        '/home/gv/school/trustworthy_ai/proj/testing_model/resume_job_pairs.csv',  # Path to pairs
        '/home/gv/school/trustworthy_ai/proj/resume_data/archive/Resume/Resume.csv',  # Path to resume data
        '/home/gv/school/trustworthy_ai/proj/job_data/job_descriptions.csv',  # Path to the job
    )

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = ResumeJobDataset(pairs_df.head(2), resume_df, job_df, tokenizer)
sample = dataset[0]


# In[10]:


print("Sample keys:", sample.keys())
print("Resume input shape:", sample['resume_input_ids'].shape)


# In[45]:


# Create train validation split

def create_train_val_dataloaders(pairs_df, resume_df, job_df, tokenizer, batch_size=32, max_length=256, train_size=0.8):
    #split into train and validation
    train_df, val_df = train_test_split(
        pairs_df,
        train_size=train_size,
        stratify=pairs_df['label'],
        random_state=42
    )

    print(f"Training set: {len(train_df)} pairs")
    print(f"Validation set: {len(val_df)} pairs")

    train_dataset = ResumeJobDataset(
        train_df, resume_df, job_df, tokenizer, max_length=max_length
    )
    val_dataset = ResumeJobDataset(
        val_df, resume_df, job_df, tokenizer, max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


# In[12]:


train_loader, val_loader = create_train_val_dataloaders(pairs_df, resume_df, job_df, tokenizer)


# In[30]:


from torch.optim import AdamW
from tqdm import tqdm


# In[31]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[38]:


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc='Training')

    for batch in progress_bar:
        optimizer.zero_grad()

        # use device
        resume_input_ids = batch['resume_input_ids'].to(device)
        resume_attention_mask = batch['resume_attention_mask'].to(device)
        job_input_ids = batch['job_input_ids'].to(device)
        job_attention_mask = batch['job_attention_mask'].to(device)
        labels = batch['label'].to(device)

        # embeddings
        resume_embeddings = model(resume_input_ids, resume_attention_mask)
        job_embeddings = model(job_input_ids, job_attention_mask)

        # calculate loss
        loss = loss_fn(resume_embeddings, job_embeddings, labels=labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(train_loader)


# In[39]:


# same as training only no backprop
def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            
            resume_input_ids = batch['resume_input_ids'].to(device)
            resume_attention_mask = batch['resume_attention_mask'].to(device)
            job_input_ids = batch['job_input_ids'].to(device)
            job_attention_mask = batch['job_attention_mask'].to(device)
            labels = batch['label'].to(device)
    
            # embeddings
            resume_embeddings = model(resume_input_ids, resume_attention_mask)
            job_embeddings = model(job_input_ids, job_attention_mask)
    
            # calculate loss
            loss = loss_fn(resume_embeddings, job_embeddings, labels=labels)
            total_loss += loss.item()

        return total_loss / len(val_loader)


# In[34]:


import os


# In[42]:


model_config = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 2e-5,
    'warmup_steps': 1000,
    'margin': 0.5,
    'train_size': 0.8,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'output_dir': 'model_output'
               }


# In[46]:


def train_model(config, pairs_df, resume_df, job_df):
    #add output dir
    os.makedirs(config['output_dir'], exist_ok=True)

    # Data loaders
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    train_loader, val_loader = create_train_val_dataloaders(
        pairs_df, resume_df, job_df, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        train_size=config['train_size']
    )

    # Initialize model
    model = ContrastiveModel(config['model_name'])
    model.to(config['device'])

    #loss function
    loss_fn = ContrastiveLoss(margin=config['margin'], loss_type='contrastive')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps)

    # Train model
    print(f"Starting training, Device: {config['device']}")
    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")

        #train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, config['device'])
        train_losses.append(train_loss)
        print(f"Training loss: {train_loss:.4f}")

        #evaluate
        val_loss = evaluate(model, val_loader, loss_fn, config['device'])
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.4f}")

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(config['output_dir'], f"checkpoint_epoch_{epoch+1}.pt"))

    # Final model
    torch.save(model.state_dict(), os.path.join(config['output_dir'], "final_model.pt"))
    # tokenizer
    tokenizer.save_pretrained(os.path.join(config['output_dir'], 'tokenizer'))

    return model, tokenizer


# In[47]:


train_model(model_config, pairs_df, resume_df, job_df)


# In[ ]:




