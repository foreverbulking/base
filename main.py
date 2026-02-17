import pymongo
import os
import datetime

uri = os.getenv("MONGO_URI") 
client = pymongo.MongoClient(uri)

db = client['jenkins-box'] 

collection = db.build_logs

data = {
    "job_name": os.getenv("JOB_NAME", "local_test"),
    "build_number": os.getenv("BUILD_NUMBER", "0"),
    "python_version": os.getenv("PYTHON_VERSION", "unknown"),
    "status": "success",
    "timestamp": datetime.datetime.now(datetime.timezone.utc)
}

collection.insert_one(data)

print(f"Successfully logged build {data['build_number']} for {data['job_name']} to MongoDB!")