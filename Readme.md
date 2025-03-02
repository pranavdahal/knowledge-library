# Q&A Knowledge Library

A Python application with Streamlit UI.

## Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- Docker (optional, for containerized deployment)

## Installation Instructions

You can set up this project using either a conda environment or a Python virtual environment.

### Option 1: Using Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/pranavdahal/knowledge-library.git
   cd knowledge-library
   ```

2. Create a new conda environment:
   ```bash
   conda create -n project-env python=3.12
   conda activate project-env
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Python Virtual Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/pranavdahal/knowledge-library.git
   cd knowledge-library
   ```

2. Create a new virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Using Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/pranavdahal/knowledge-library.git
   cd knowledge-library
   ```

2. Build the Docker image:
   ```bash
   docker build -t knowledge-library .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8501:8501 -d knowledge-library
   ```

   The application will be available at http://localhost:8501 in your web browser.

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
NEO4J_PASSWORD=your_value_here
NEO4J_URL=your_value_here
NEO4J_USERNAME=your_value_here
```

Replace `your_value_here` with the appropriate values.

### For Docker

You can pass environment variables to the Docker container using the `-e` flag:

```bash
docker run -p 8501:8501 -e VARIABLE_1=value1 -e VARIABLE_2=value2 -e VARIABLE_3=value3 -d knowledge-library
```

Alternatively, you can use an environment file:

```bash
docker run -p 8501:8501 --env-file .env -d knowledge-library
```

## Running the Application

### Local Deployment

To run the application locally:

```bash
streamlit run Home.py
```

The application will be available at http://localhost:8501 in your web browser.

## Project Structure

```
knowledge-library/
├── Home.py                # Main entry point for the application
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (create this file)
└── ...                    # Other project files
```
