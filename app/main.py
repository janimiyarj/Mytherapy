from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from app.openai_agent import generate_advice_with_rag
from app.model import predict_response_type, cluster_to_category
from rag.rag_engine import retrieve_relevant_chunks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# CORS Setup – for local/frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod: restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESPONSE_TYPE_LABELS = {
    0: "Mild Distress",
    1: "Moderate Distress",
    2: "Severe Distress",
    3: "Positive / Recovery",
}

# Pydantic Input Schema
class PatientInput(BaseModel):
    description: str

# Default Route (for browser view)
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>MYTHERAPY — Mental Health Classification and Chat Support App</title></head>
        <body>
            <h2>Mental Health Assistant</h2>
            <p>POST to <code>/predict</code> or <code>/advise</code> to receive support.</p>
        </body>
    </html>
    """

@app.post("/predict")
async def predict_response(data: PatientInput):
    logger.info(f"Received predict request: {data.description}")
    try:
        prediction = predict_response_type(data.description)
        label = cluster_to_category.get(int(prediction), "Unknown")
        logger.info(f"Prediction result: {prediction} → {label}")
        return {
            "predicted_cluster_id": int(prediction),
            "category": label
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)




# Smart RAG Advice (used by frontend "Get AI Advice" button)
@app.post("/advise")
async def get_advice(data: PatientInput):
    logger.info(f"Received advice request: {data.description}")
    try:
        if not data.description.strip():
            logger.warning("Empty description received.")
            return JSONResponse(content={"error": "No input description provided."}, status_code=400)

        advice = generate_advice_with_rag(data.description)
        logger.info("Generated advice successfully.")
        return {"advice": advice}
    except Exception as e:
        logger.error(f"Advice generation error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# View Retrieved RAG Chunks (Optional for debugging/tools)
@app.post("/rag-advise")
async def get_contextual_advice(data: PatientInput):
    try:
        logger.info("Running RAG pipeline...")
        chunks = retrieve_relevant_chunks(data.description, top_k=3)
        logger.info("RAG completed.")
        return {"retrieved_chunks": chunks}
       
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
