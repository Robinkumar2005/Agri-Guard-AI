from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["agriguard"]

predictions_collection = db["predictions"]
conversations_collection = db["conversations"]
