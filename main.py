import pymongo
import os

# 1. Connect to the Cluster
uri = os.getenv("MONGO_URI") 
client = pymongo.MongoClient(uri)

# 2. Access your specific database
db = client['jenkins-box'] # Updated to match your name

# 3. Access a collection (like a folder for your records)
collection = db.build_logs

# 4. Insert data
data = {"job": "maple-run", "status": "success"}
collection.insert_one(data)

print("Successfully sent data to jenkins-box database!")