import pymupdf
from llama_cpp import Llama
import os
from langchain_ollama import OllamaLLM


def improve_prompt(text):
    """Create a more effective prompt that will get better tagging results."""
    improved_prompt = f"""
    Task: Tag each word in the resume with an entity label following BIO (Beginning, Inside, Outside) format.
    
    Instructions:
    1. Process the resume word-by-word, assigning tags to EVERY word.
    2. Use these tags exactly:
       - [B-HEADER] - Beginning of section headers
       - [I-HEADER] - Continuation of section headers
       - [B-EXP] - Beginning of job experience entries
       - [I-EXP] - Continuation of job experience entries
       - [B-EDU] - Beginning of education entries
       - [I-EDU] - Continuation of education entries
       - [B-SKILLS] - Beginning of skills section
       - [I-SKILLS] - Continuation of skills
       - [B-OBJECTIVE] - Beginning of career objective
       - [I-OBJECTIVE] - Continuation of career objective
    
    IMPORTANT: Tag EVERY word. No exceptions. Tag punctuation with its preceding word.
    
    Example of correct tagging:
    ACCOUNTANT[B-HEADER]
    Summary[B-HEADER]
    Financial[B-OBJECTIVE] Accountant[I-OBJECTIVE] specializing[I-OBJECTIVE] in[I-OBJECTIVE] financial[I-OBJECTIVE] planning,[I-OBJECTIVE] reporting[I-OBJECTIVE] and[I-OBJECTIVE] analysis[I-OBJECTIVE] within[I-OBJECTIVE] the[I-OBJECTIVE] Department[I-OBJECTIVE] of[I-OBJECTIVE] Defense.[I-OBJECTIVE]
    
    Now tag the following resume with ONLY the tagged version:
    {text}
    """
    return improved_prompt


def get_response_with_ollama(text, ):
    prompt = f"""
    Task: Tag each word in the resume with an entity label following BIO (Beginning, Inside, Outside) format.
    Instructions:
    1. Process the resume word-by-word, assigning tags to EVERY word.
    2. Use the following tag schema exactly:
       - [B-HEADER] - Beginning of section headers (e.g., "EXPERIENCE", "EDUCATION")
       - [I-HEADER] - Continuation of section headers
       - [B-EXP] - Beginning of job experience entries (position or company)
       - [I-EXP] - Continuation of job experience entries (details, dates, bullet points)
       - [B-EDU] - Beginning of education entries
       - [I-EDU] - Continuation of education entries
       - [B-SKILLS] - Beginning of skills section
       - [I-SKILLS] - Continuation of skills
       - [B-OBJECTIVE] - Beginning of career objective
       - [I-OBJECTIVE] - Continuation of career objective
       
    Important rules:
    - Every word MUST have a tag - no exceptions
    - For bullet points (•, -, *), tag as [I-EXP] in experience sections
    - Tag punctuation marks together with their preceding word
    - Be consistent with tagging bullets/hyphens as [I-EXP] when they appear in experience sections
    - For dates, always use the Inside tag of the corresponding section (e.g., [I-EXP], [I-EDU])
    
    Example:
    
    Original:
    ENGINEERING LAB TECHNICIAN
    Career Focus
    My main objective in seeking employment with Triumph Actuation Systems Inc. is to work in a professional atmosphere where I can utilize my skills and continue to gain experience in the aerospace industry to advance in my career.
    Professional Experience
    Engineering Lab Technician Oct 2016 to Current
    Company Name – City, State
    - Responsible for testing various seat structures to meet specific certification requirements.
    
    Tagged:
    ENGINEERING[B-HEADER] LAB[I-HEADER] TECHNICIAN[I-HEADER]
    Career[B-HEADER] Focus[I-HEADER]
    My[B-OBJECTIVE] main[I-OBJECTIVE] objective[I-OBJECTIVE] in[I-OBJECTIVE] seeking[I-OBJECTIVE] employment[I-OBJECTIVE] with[I-OBJECTIVE] Triumph[I-OBJECTIVE] Actuation[I-OBJECTIVE] Systems[I-OBJECTIVE] Inc.[I-OBJECTIVE] is[I-OBJECTIVE] to[I-OBJECTIVE] work[I-OBJECTIVE] in[I-OBJECTIVE] a[I-OBJECTIVE] professional[I-OBJECTIVE] atmosphere[I-OBJECTIVE] where[I-OBJECTIVE] I[I-OBJECTIVE] can[I-OBJECTIVE] utilize[I-OBJECTIVE] my[I-OBJECTIVE] skills[I-OBJECTIVE] and[I-OBJECTIVE] continue[I-OBJECTIVE] to[I-OBJECTIVE] gain[I-OBJECTIVE] experience[I-OBJECTIVE] in[I-OBJECTIVE] the[I-OBJECTIVE] aerospace[I-OBJECTIVE] industry[I-OBJECTIVE] to[I-OBJECTIVE] advance[I-OBJECTIVE] in[I-OBJECTIVE] my[I-OBJECTIVE] career.[I-OBJECTIVE]
    Professional[B-HEADER] Experience[I-HEADER]
    Engineering[B-EXP] Lab[I-EXP] Technician[I-EXP] Oct[I-EXP] 2016[I-EXP] to[I-EXP] Current[I-EXP]
    Company[I-EXP] Name[I-EXP] –[I-EXP] City,[I-EXP] State[I-EXP]
    - [I-EXP] Responsible[I-EXP] for[I-EXP] testing[I-EXP] various[I-EXP] seat[I-EXP] structures[I-EXP] to[I-EXP] meet[I-EXP] specific[I-EXP] certification[I-EXP] requirements.[I-EXP]
    
    Now tag the following resume using the exact same format and rules. Return ONLY the tagged version:
    
    {text}
    """
    
    llm = OllamaLLM(
        model="llama3.1:8b",  
        temperature=0.1,
        top_p=0.95
    )
    prompt = improve_prompt(text)
    
    # Prompt needs to be improved, sometimes works sometimes doesnt
    response = llm.invoke(prompt)
    return response.strip()

def process_resume(resume_paths, output_dir=None):
    for path in resume_paths:
        doc = pymupdf.open(path)
        text = chr(12).join([page.get_text() for page in doc])

        tagged_resume = get_response_with_ollama(text)

        output_path = os.path.splitext(path)[0] + 'tagged.txt'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tagged_resume)

def get_files(root_dir):
    files_by_category = {}
    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))]
    
    for category in categories:
        category_path = os.path.join(root_dir,category)

        pdf_files = [
            os.path.join(category_path,f)
            for f in os.listdir(category_path)
            if f.lower().endswith('.pdf')
        ] 
     
        files_by_category[category] = pdf_files
        #print(pdf_files)
        #print(files_by_category)
    return files_by_category


folder_path = '/home/gv/school/trustworthy_ai/proj/test_reading_files/'

if __name__ == "__main__":
    #resume_path = "natalia-resume.pdf"
    files_by_category = get_files(folder_path)
    for category, file_paths in files_by_category.items():
        tagged_resume = process_resume(file_paths)
