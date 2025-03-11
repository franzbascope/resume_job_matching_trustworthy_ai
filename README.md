# AI-Powered Resume-to-Job Matching

## Project Overview
This project uses semantic similarity to match resumes with relevant job descriptions. We fine-tune SBERT (Sentence-BERT) and leverage pre-trained embeddings to improve job matching accuracy. The goal is to create an AI system that enhances hiring efficiency by finding the best-fit jobs for candidates based on their skills and experience.

## Features
- **Resume and Job Description Matching:** Uses natural language processing (NLP) techniques to assess compatibility.
- **Fair and Explainable AI:** Implements techniques to ensure recommendations are interpretable and free from bias.
- **Web Scraping for Job Listings:** Automates job posting retrieval for a more comprehensive dataset.
- **Fine-Tuned AI Models:** Utilizes machine learning to enhance accuracy in job matching.
- **User-Friendly API:** Provides endpoints for easy integration with external applications.

## Project Structure
```
resume_job_matching_trustworthy_ai/
│── backend/
│── dataset/
│── final_model/
│── job_web_scraping/
│── jupyter_notebooks/
│── samples/
│── .gitattributes
│── .gitignore
│── README.md
```

## Prerequisites
- Python 3.8+
- pip
- Virtual environment (optional)
  
## Setup & Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/franzbascope/resume_job_matching_trustworthy_ai.git
cd resume_job_matching_trustworthy_ai
```

### **2. Create and activate a virtual environment**
```bash
python3 -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate'
```

### **3. Install the required dependencies**
```bash
pip install -r requirements.txt
```

## How to Use
1. Upload a Resume
2. Run Matching Algorithm
3. Review Results

