from flask import Flask, request, jsonify, flash, redirect, url_for, session
from flask import render_template
import fitz  # PyMuPDF
import logging
import random

app = Flask(__name__)
app.secret_key = '6durcsus!zdgr&9%4$*(^kttr7%pm(x=3o!&*px^uev)sg)ssp'

# Configure logging
logging.basicConfig(level=logging.DEBUG)
ALLOWED_EXTENSIONS = {"pdf"}


@app.route('/')
def upload():
    return render_template('compare-resume.html')


@app.route('/results', methods=['GET'])
def results():
    results = session.get('resume_results')
    return render_template('results.html',results=results)


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

    resume_text = extract_text_from_pdf(resume)
    job_listings_texts = []
    for job_file in job_files:
        job_text = extract_text_from_pdf(job_file)
        job_listings_texts.append({
            "fileName": job_file.filename,
            "content": job_text
        })
    scores = [calculate_similarity(resume_text, job_text)
              for job_text in job_listings_texts]
    session['resume_results'] = scores
    return redirect(url_for('results'))


@app.route("/match", methods=["POST"])
def match_resume():
    if 'resume' not in request.files:
        app.logger.error("Missing resume file")
        return jsonify({"error": "Missing resume file"}), 400
    if 'job_listings' not in request.files:
        app.logger.error("Missing job listings files")
        return jsonify({"error": "Missing job listings files"}), 400

    resume_file = request.files['resume']
    job_listings_files = request.files.getlist('job_listings')

    if not resume_file:
        app.logger.error("Resume file is empty")
        return jsonify({"error": "Resume file is empty"}), 400
    if not job_listings_files:
        app.logger.error("Job listings files are empty")
        return jsonify({"error": "Job listings files are empty"}), 400

    resume_text = extract_text_from_pdf(resume_file)
    job_listings_texts = []
    for job_file in job_listings_files:
        job_text = extract_text_from_pdf(job_file)
        job_listings_texts.append({
            "fileName": job_file.filename,
            "content": job_text
        })

    # Here you would implement your matching logic
    # For demonstration, let's assume we have a function `calculate_similarity`
    scores = [calculate_similarity(resume_text, job_text)
              for job_text in job_listings_texts]

    return jsonify({"scores": scores})


def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def calculate_similarity(resume_text, job_dic):
    # generate a random number with 2 decimals from 0 to 100
    rand = round(random.uniform(0, 100), 2)
    return {
        "file_name": job_dic["fileName"],
        "score": rand
    }

if __name__ == "__main__":
    app.run(debug=True)
