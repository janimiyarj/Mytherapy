#  MYTHERAPY — Mental Health Classification and Chat Support App

MYTHERAPY is an intelligent mental health support system that classifies user inputs into therapeutic categories (like *Directive*, *Emotional*, *Supportive*, etc.) using a machine learning model trained on real therapy data. It also integrates a Retrieval-Augmented Generation (RAG) chatbot to provide personalized, secure responses.

---

##  Project Structure

```
MYTHERAPY/
├── app/                        # Backend logic (FastAPI/Flask-ready)
│   ├── __init__.py
│   ├── main.py                 # Main app entry point
│   ├── model.py                # Model loading and prediction
│   └── openai_agent.py         # OpenAI agent for RAG response
│
├── data/
│   └── train.csv               # Original training data
│
├── frontend/
│   └── index.html              # Simple front-end for user interaction
│
├── model/
│   ├── classifier.pkl          # Trained KMeans classification model
│   └── vectorizer.pkl          # TF-IDF vectorizer used for feature extraction
│
├── notebooks/
│   └── data_analysis.ipynb     # Model training and clustering notebook
│
├── rag/
│   ├── rag_builder.py          # Prepares vector store for RAG
│   ├── rag_engine.py           # Embedding-based retrieval + chatbot response
│   └── vectors/
│       ├── faiss_index.bin
│       └── metadata.json
│
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── ReadMe.md                  # You're here!
```



---

##  Features

- **ML Model for Classification**  
  - Categories: `Directive`, `Emotional`, `Supportive`, `Cognitive`, `Neutral`, `Other`
  - Uses TF-IDF + KMeans clustering
  - Trained on anonymized therapy conversations

- **Retrieval-Augmented Generation (RAG)**  
  - FAISS + OpenAI GPT + metadata-based chunk search  
  - Enhances response quality by grounding in real examples

- **Lightweight Frontend**  
  - Basic `index.html` to send and view messages

- **Modular Codebase**  
  - Clear separation of logic for modeling, RAG, and serving

---

##  Setup Instructions

### 1. Clone the Repository
bash: 

git clone https://github.com/your-username/mytherapy.git

cd mytherapy

### 2. Create Virtual Environment & Install Dependencies

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

### 3. Train Model (Optional — already trained)

jupyter notebook notebooks/data_analysis.ipynb

### 4. Start Backend

cd app

uvicorn main:app --reload


### 5. Open Frontend

Open frontend/index.html in a browser.

### 6. Model Details
### Vectorizer: TF-IDF

Model: KMeans clustering

#### Clusters Mapped As:

cluster_to_category = {
    0: "Directive",
    1: "Emotional",
    2: "Supportive",
    3: "Cognitive",
    4: "Neutral",
    5: "Other"
}


#### Environment Variables

Create a .env file and add:

Code:  
   OPENAI_API_KEY=your_openai_api_key

### 7. Dependencies
Install via:

pip install -r requirements.txt

Includes:

scikit-learn

pandas

faiss-cpu

openai

fastapi, uvicorn

joblib

jinja2

### 8. Future Improvements
Deploy on cloud (e.g. Render, AWS, GCP, Azure)

Improve frontend with React or Svelte

Add sentiment analysis pipeline

Extend to multilingual therapy

Integrate MLOps best practices:

Use MLflow or DVC for model versioning and experiment tracking

Set up CI/CD pipelines with GitHub Actions or GitLab CI

Containerize the app with Docker and orchestrate with Kubernetes

Use Terraform or Pulumi for infrastructure as code (IaC)

Monitor model performance using Prometheus and Grafana

Automate retraining with cloud workflows (e.g., AWS Step Functions, GCP Workflows)

### Contributors
Jani Miya Shaik — ML Engineering, Backend, Chatbot Design

Collaborators welcome!

## License
MIT License — feel free to use, extend, and contribute.
