# Job Matching System

This project is a Flask-based application that allows users to upload a resume and receive the best-matching job listings based on NLP-based job matching.

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Create a Virtual Environment
To ensure dependency isolation, create a virtual environment:
```sh
python -m venv venv
```

### 2. Activate the Virtual Environment
Activate the virtual environment depending on your operating system:
- **Windows**:
  ```sh
  venv\Scripts\activate
  ```
- **MacOS/Linux**:
  ```sh
  source venv/bin/activate
  ```

### 3. Install Dependencies
With the virtual environment activated, install the required dependencies:
```sh
pip install -r ./backend/requirements.txt
```

### 4. Install Git Large File Storage (LFS)
This project requires Git LFS to manage large files. Install it using:
- **Windows**:
  ```sh
  choco install git-lfs
  ```
- **MacOS**:
  ```sh
  brew install git-lfs
  ```
- **Linux**:
  ```sh
  sudo apt install git-lfs  # Debian/Ubuntu
  sudo dnf install git-lfs  # Fedora
  ```
After installation, enable Git LFS:
```sh
git lfs install
```

### 5. Pull Large Files from Git LFS
Ensure large files are retrieved by running:
```sh
git lfs pull
```

### 6. Initialize the Backend (Run Once)
Before starting the server for the first time, initialize the backend by executing:
```sh
python backend/init.py
```

### 7. Start the Server
Run the following command to start the Flask server:
```sh
python backend/main.py
```

### 8. Access the Application
Once the server is running, open your browser and go to:
```
http://127.0.0.1:5000
```

The application should load successfully, allowing you to upload a resume and receive job matches.

