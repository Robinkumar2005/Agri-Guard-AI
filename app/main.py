import io
from io import BytesIO
import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Annotated
import time
from PIL import Image
from app.services.classifier import get_model, single_predict as single, batch_predict as batch
from app.database.mongodb import predictions_collection, conversations_collection
from datetime import datetime, timezone
from app.services.agent import get_agent_response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from bson import ObjectId
from fastapi.openapi.utils import get_openapi



app = FastAPI(
    title="Agriculture Image Classification API",
    description="An API for classifying agricultural images using a pre-trained model.",
    version="1.0.0"
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Use OpenAPI 3.0.3 for Swagger file upload compatibility
    schema["openapi"] = "3.0.3"

    components = schema.get("components", {}).get("schemas", {})

    for component in components.values():
        properties = component.get("properties", {})

        for prop in properties.values():

            # Fix multiple file uploads
            if prop.get("type") == "array":
                items = prop.get("items", {})

                if items.get("contentMediaType") == "application/octet-stream":
                    items.pop("contentMediaType", None)
                    items["format"] = "binary"

            # Fix single file uploads
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/Agri_Guard_api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_request(request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        logger.info(
            f"Request: {request.method} {request.url} "
            f"completed in {process_time:.4f}s "
            f"with status code {response.status_code}"
        )
        return response
    except Exception:
        logger.exception("Error occurred while processing the request.")
        raise
    
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the Agriculture Image Classification API...")
    try:
        get_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load the model: {e}")
        raise RuntimeError("Model loading failed. Check logs for details.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the Agriculture Image Classification API...")

@app.get("/")
def home():
    logger.info("Home endpoint accessed.")
    return {"message": "Welcome to the Agriculture Image Classification API!"}

@app.get("/model_info")
def model_info():
    logger.info("Model info endpoint accessed.")
    return{
        "model_name": "AgriGuard",
        "version": "1.0.0",
        "description": "A trained model for classifying agricultural images into various categories."
    }
@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed.")
    return {
        "status": "healthy",
        "model_loaded":get_model() is not None,
        "message": "The API is running smoothly.",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/single_predict")
async def single_predict(file: UploadFile = File(...)):
    logger.info(f"Single prediction request received for file: {file.filename}")
    try:
        image = Image.open(BytesIO(await file.read()))
        prediction = single(image)
        agent_response = await run_in_threadpool(
            get_agent_response,
            prediction["disease"],
            prediction["confidence"]
        )
        prediction_data = {
        "filename": file.filename,
        "disease": prediction["disease"],
        "confidence": prediction["confidence"],
        "ai_explanation": agent_response,
        "created_at": datetime.now(timezone.utc)
    }
        
        result = predictions_collection.insert_one(prediction_data)
        logger.info(f"Prediction completed for file: {file.filename} with result: {prediction}")
        return {
            "filename": file.filename,
            "prediction": prediction,
            "ai_explanation": agent_response,
            "database_id": str(result.inserted_id)
        }    
        
    except Exception as e:
        logger.error(f"Error during prediction for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check logs for details.")

@app.post("/batch_predict")
async def batch_predict(files:List[UploadFile] = File(...)) -> List[dict]:
    logger.info(f"Batch prediction request received for {len(files)} files.")
    predictions = []
    for file in files:
        try:
            image = Image.open(BytesIO(await file.read()))
            prediction = batch([image])[0]
            agent_response = await run_in_threadpool(
                get_agent_response,
                prediction["disease"],
                prediction["confidence"]
            )
            prediction_data = {
                "filename": file.filename,
                "disease": prediction["disease"],
                "confidence": prediction["confidence"],
                "ai_explanation": agent_response,
                "created_at": datetime.now(timezone.utc)
            }
            result = predictions_collection.insert_one(prediction_data)
            predictions.append({
                "filename": file.filename,
                "prediction": prediction,
                "ai_agent": {
                    "name": "AgriGuard",
                    "response": agent_response
                },
                "database_id": str(result.inserted_id)
            })
            logger.info(f"Prediction completed for file: {file.filename} with result: {prediction}")
        except Exception as e:
            logger.error(f"Error during prediction for file {file.filename}: {e}")
            predictions.append({
                "filename": file.filename,
                "error": "Prediction or AI explanation failed. Check logs for details."
            })
    return predictions

            

class ChatRequest(BaseModel):
    prediction_id: str
    message: str

@app.post("/chat")
async def chat_with_agriguard(request: ChatRequest):

    try:
        if not ObjectId.is_valid(request.prediction_id):
            raise HTTPException(status_code=400,detail="Invalid prediction ID.")

        if not request.message.strip():
            raise HTTPException(status_code=400,detail="Message cannot be empty.")

        prediction = predictions_collection.find_one({
            "_id": ObjectId(request.prediction_id)
        })

        if prediction is None:
            raise HTTPException(status_code=404,detail="Prediction not found.")

        agent_response = await run_in_threadpool(
            get_agent_response,
            prediction["disease"],
            prediction["confidence"],
            request.message
        )

        conversation_data = {
            "prediction_id": prediction["_id"],
            "user_message": request.message,
            "ai_response": agent_response,
            "created_at": datetime.now(timezone.utc)
        }

        result = conversations_collection.insert_one(
            conversation_data
        )

        logger.info("AgriGuard conversation saved successfully.")

        return {
            "agent": "AgriGuard",
            "prediction_id": request.prediction_id,
            "user_message": request.message,
            "ai_response": agent_response,
            "conversation_id": str(result.inserted_id)
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"AgriGuard chat failed: {e}")

        raise HTTPException(status_code=500,detail="AgriGuard could not process your message.")
