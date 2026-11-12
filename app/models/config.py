import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH = os.getenv("Model_Path", "models/best_model.pth")
    CLASS_PATH = os.getenv("Class_Path", "models/class_names.json")
settings = Settings()
