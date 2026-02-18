import os
import pymongo
import certifi


class MockDB:
    def __init__(self):
        # Use the URI we set up for your local Jenkins/Mongo test
        uri = os.getenv("MONGO_URI")
        self.client = pymongo.MongoClient(uri, tlsCAFile=certifi.where())
        self.db = self.client["jenkins-box"]  # Matches your local DB name


def get_db():
    return MockDB()
