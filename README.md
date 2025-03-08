# AI-Powered Resume-to-Job Matching

## Project Overview
This project uses semantic similarity to match resumes with relevant job descriptions. We fine-tune SBERT (Sentence-BERT) and leverage pre-trained embeddings to improve job matching accuracy. The goal is to create an AI system that enhances hiring efficiency by finding the best-fit jobs for candidates based on their skills and experience.

## Features
- **Resume-to-Job Matching**: Uses SBERT to compute similarity scores between resumes and job descriptions.
- **Fine-Tuned Model**: Improves baseline performance by training on curated resume-job pairs.
- **API Integration**: Exposes a RESTful API for querying job matches.
- **Trustworthy AI Focus**: Ensures fairness in matching and avoids bias.

## 📂 Project Structure
```
resume_job_matching_trustworthy_ai/
│── creating_job_resume_pairs.ipynb   # Prepares training data (positive resume-job pairs)
│── llama_8b_test.py                  # Initial test using LLaMA for resume tagging
│── resume_job_matching.ipynb         # Main notebook for model training and evaluation
│── README.md                         # Project documentation
```

## Setup & Installation
### **1 Clone the Repository**
```bash
git clone https://github.com/franzbascope/resume_job_matching_trustworthy_ai.git
cd resume_job_matching_trustworthy_ai
```

### **2 Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3 Run Model Training (Optional, If Fine-Tuning Again)**
```bash
python resume_job_matching.ipynb
```

### **4 Run API for Job Matching**
```bash
cd api
python app.py
```

## How to Use
1. Upload resumes and job descriptions.
2. The model computes similarity scores.
3. Results show the best job match for each resume.

