import os
import atexit
import coverage
import pymongo

job_name = os.environ.get("JOB_NAME")
build_id = os.environ.get("BUILD_NUMBER")
python_version = os.environ.get("PYTHON_VERSION") 

def should_run_coverage():
    if not all([job_name, python_version]):
        return False 
    
    uri = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(uri)
    db = client['jenkins-box']
    
    # Check if this specific job/version combo has a baseline.
    exists = db.coverage_registry.find_one({
        "job_name": job_name, 
        "python_version": python_version
    })
    return exists is None

if should_run_coverage():
    output_dir = os.path.join("coverage", job_name, str(build_id))
    os.makedirs(output_dir, exist_ok=True)
    
    data_file = os.path.join(output_dir, ".coverage")
    cov = coverage.Coverage(data_file=data_file)
    cov.start()

    @atexit.register
    def stop_coverage():
        cov.stop()
        cov.save()
        cov.xml_report(outfile=os.path.join(output_dir, "coverage.xml"))
        
        # Register it so we never run it again.
        uri = os.getenv("MONGO_URI")
        client = pymongo.MongoClient(uri)
        client['jenkins-box'].coverage_registry.insert_one({
            "job_name": job_name,
            "python_version": python_version,
            "build_id": build_id,
            "xml_path": os.path.join(output_dir, "coverage.xml")
        })
else:
    print(f"Baseline already exists for {job_name} on {python_version}. Skipping coverage.")