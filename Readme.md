# Q&A Knowledge Library

A Q&A Knowledge Library application with Streamlit UI.

## Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- Docker (optional, for containerized deployment) (preferred)
## Installation Instructions

You can set up this project using either a conda environment or a Python virtual environment.

### Option 1: Using Docker Compose

1. Clone the repository:
   ```bash
   git clone https://github.com/pranavdahal/knowledge-library.git
   cd knowledge-library
   ```

2. Build and start the containers:
   ```bash
   docker compose up --build -d
   ```

3. To stop containers:
   ```bash
   docker compose down
   ```

### Option 2: Using Conda (Run Locally)

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

### Option 3: Using Python Virtual Environment

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

### Option 4: Using Docker for streamlit and neo4j cloud

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

   You can pass environment variables to the Docker container using the `-e` flag:

   ```bash
   docker run -p 8501:8501 -e NEO4J_PASSWORD=value1 -e NEO4J_URL=value2 -e NEO4J_USERNAME=value3 -d knowledge-library
   ```

   Alternatively, you can use an environment file:

   ```bash
   docker run -p 8501:8501 --env-file .env -d knowledge-library
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

<br>

### Neo4J local Setup (using Docker)
Pull the image of neo4j from dockerhub and run your container.
```
docker run -d \
  --name neo4j \
  --publish 7474:7474 --publish 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j
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
