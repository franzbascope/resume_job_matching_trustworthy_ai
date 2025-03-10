from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>He he he ha ha ha hi hi hi!</p>"

"""
need to create a route with post method that can receive a resume pdf and 
job listings pdfs and returns the score of similarity or best match for each 
job listing.

The Request Object
https://flask.palletsprojects.com/en/stable/quickstart/#the-request-object


"""