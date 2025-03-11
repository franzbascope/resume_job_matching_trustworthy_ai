from flask import Flask, request, jsonify, flash, redirect, url_for, session
import torch
from flask import render_template
import fitz  # PyMuPDF
import logging
import random
import os
import sys
import pandas as pd
from transformers import AutoTokenizer
from scipy.spatial.distance import cosine

from contrastive_model import ContrastiveModel
app = Flask(__name__)
app.secret_key = '6durcsus!zdgr&9%4$*(^kttr7%pm(x=3o!&*px^uev)sg)ssp'

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

# Load model, tokenizer and job data
def initialize():
    model = ContrastiveModel(model_config['model_name'])

    model_path = '/home/gv/school/trustworthy_ai/proj/resume_job_matching_trustworthy_ai/final_model/final_model.pt'
    model.load_state_dict(torch.load(model_path, map_location=model_config['device']))
    model.to(model_config['device'])

    tokenizer = AutoTokenizer.from_pretrained('/home/gv/school/trustworthy_ai/proj/resume_job_matching_trustworthy_ai/final_model/tokenizer/')

    job_df = pd.read_csv('/home/gv/school/trustworthy_ai/proj/job_data/job_descriptions.csv')

    return model, tokenizer, job_df
   
model, tokenizer, job_df = initialize()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
ALLOWED_EXTENSIONS = {"pdf"}


@app.route('/')
def upload():
    return render_template('compare-resume.html')


@app.route('/results', methods=['GET'])
def results():
    results = session.get('resume_results')
    return render_template('results.html', results=results)


def allowed_file(filename):
    """Check if file has a valid PDF extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload/resume", methods=["POST"])
def upload_resume():
    """Handle file uploads."""
    # Handle resume upload
    resume = request.files.get("resume")
    job_files = request.files.getlist("job_files")  # Get multiple job listings

    # Validate resume
    if not resume or not allowed_file(resume.filename):
        flash("Invalid resume file! Only PDFs are allowed.", "danger")
        return redirect(url_for("upload"))

    # Validate job listings
    if not job_files or any(not allowed_file(job.filename) for job in job_files):
        flash("Invalid job listing files! Only PDFs are allowed.", "danger")
        return redirect(url_for("upload"))

    try:
        resume_text = extract_text_from_pdf(resume)
        
        # Get recommendations from job database
        top_n = 5  
        recommendations = recommend_jobs(resume_text, job_df, model, tokenizer, model_config, top_n=top_n)
        
        # Convert to list for session storage
        results = recommendations.to_dict('records')
        session['resume_results'] = results
        
        return redirect(url_for('results'))
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        flash(f"Error processing files: {str(e)}", "danger")
        return redirect(url_for("upload"))


@app.route("/match", methods=["POST"])
def match_resume():
    if 'resume' not in request.files:
        app.logger.error("Missing resume file")
        return jsonify({"error": "Missing resume file"}), 400

    resume_file = request.files['resume']

    if not resume_file:
        app.logger.error("Resume file is empty")
        return jsonify({"error": "Resume file is empty"}), 400

    top_n = request.args.get('top_n', default=5, type=int)

    try:
        resume_text = extract_text_from_pdf(resume_file)
        app.logger.debug(f"Extracted {len(resume_text)} characters from resume")

        # Get recommendations
        recommendations = recommend_jobs(resume_text, job_df, model, tokenizer, model_config, top_n=top_n)

        results = recommendations.to_dict('records')

        return jsonify({
            "matches": results,
            "total_jobs_compared": len(job_df)
        })
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


def recommend_jobs(resume_text, job_df, model, tokenizer, config, top_n=5):
    resume_embedding = get_resume_embedding(resume_text, model, tokenizer, config) 
    similarities = []

    batch_size = 100

    for i in range(0, len(job_df), batch_size):
        batch_df = job_df.iloc[i:i+batch_size]

        for _, row in batch_df.iterrows():
            job_id = row['Job Id']
            job_title = row['Job Title']
            job_desc = row['Job Description']
            job_text = f"{job_title}. {job_desc}"

            job_embedding = get_job_embedding(job_text, model, tokenizer, config)

            #cosine similarity
            similarity = 1 - cosine(resume_embedding, job_embedding)

            location = row.get('location', 'Not specified')
            company = row.get('Company', 'Not specified')

            similarities.append({
                'job_id': job_id,
                'similarity': float(similarity),
                'job_title': job_title,
                'company': company,
                'location': location
            })

    similarities_df = pd.DataFrame(similarities)
    similarities_df = similarities_df.sort_values('similarity', ascending=False)

    return similarities_df.head(top_n)


def get_resume_embedding(resume_text, model, tokenizer, config):
    # embedding of single resume
    encoding = tokenizer.encode_plus(
            resume_text,
            max_length=config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'  
        )

    input_ids = encoding['input_ids'].to(config['device'])
    attention_mask = encoding['attention_mask'].to(config['device'])

    model.eval()
    with torch.no_grad():
        embedding = model.get_embeddings(input_ids, attention_mask)

    return embedding.cpu().numpy()


def get_job_embedding(job_text, model, tokenizer, config):
    # get embedding for single job
    encoding = tokenizer.encode_plus(
            job_text,
            max_length=config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'  
        )
    input_ids = encoding['input_ids'].to(config['device'])
    attention_mask = encoding['attention_mask'].to(config['device'])

    model.eval()
    with torch.no_grad():
        embedding = model.get_embeddings(input_ids, attention_mask)

    return embedding.cpu().numpy()


def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


if __name__ == "__main__":
    app.run(debug=True)
