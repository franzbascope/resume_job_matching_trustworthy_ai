from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
    job_listings_texts = [extract_text_from_pdf(job_file) for job_file in job_listings_files]

    # Here you would implement your matching logic
    # For demonstration, let's assume we have a function `calculate_similarity`
    scores = [calculate_similarity(resume_text, job_text) for job_text in job_listings_texts]

    return jsonify({"scores": scores})

def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def calculate_similarity(resume_text, job_text):
    # Implement your similarity calculation logic here
    # For demonstration, let's return a dummy score
    return 0.5

if __name__ == "__main__":
    app.run(debug=True)