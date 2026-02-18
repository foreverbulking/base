"""
Runtime coverage collector for Jenkins jobs.
"""

import os
import sys
import platform
import atexit


def log_coverage(message):
    print(f"[coverage] {message}", file=sys.stderr, flush=True)


def _should_run():
    """Check if coverage should run."""
    if "JENKINS_URL" not in os.environ:
        return False

    if "AAA_BASE" not in os.environ:
        return False

    return True


def _main():
    if not _should_run():
        log_coverage("skipped: not running on Jenkins")
        return

    job_name = os.environ.get("JOB_NAME", "unknown")
    python_version = platform.python_version()
    repo_base_path = os.path.abspath(os.environ["AAA_BASE"])
    registry_key = f"{job_name}|{python_version}"

    # Check if coverage already exists
    try:
        if repo_base_path not in sys.path:
            sys.path.insert(0, repo_base_path)

        import utils.dal.dal as dal

        registry_collection = dal.get_db().db["coverage_registry"]

        # Skip if already recorded
        if registry_collection.find_one({"_id": registry_key}, {"_id": 1}):
            log_coverage(f"skipped: {job_name} already recorded for {python_version}")
            return

    except Exception as e:
        log_coverage(f"skipped: cannot reach MongoDB for {job_name} ({e})")
        return

    # Start coverage
    try:
        import coverage
        import json
        import tempfile
        import datetime

        coveragerc_path = os.path.join(repo_base_path, ".coveragerc")
        coverage_instance = coverage.Coverage(
            config_file=coveragerc_path, source=[repo_base_path]
        )
        coverage_instance.start()
        log_coverage(f"started for {job_name}")

        def _stop_coverage():
            try:
                coverage_instance.stop()
                coverage_instance.save()

                # Write to a unique temp file to avoid collisions
                file_descriptor, temp_json_path = tempfile.mkstemp(suffix=".json")
                try:
                    coverage_instance.json_report(
                        outfile=temp_json_path, pretty_print=False
                    )
                    with os.fdopen(file_descriptor, "r") as json_file:
                        coverage_data = json.load(json_file)
                finally:
                    if os.path.exists(temp_json_path):
                        os.remove(temp_json_path)

                coverage_percent = coverage_data.get("totals", {}).get(
                    "percent_covered", 0.0
                )

                # Store results
                registry_collection.insert_one(
                    {
                        "_id": registry_key,
                        "job_name": job_name,
                        "python_version": python_version,
                        "collected_at": datetime.datetime.now(datetime.timezone.utc),
                        "coverage_percent": coverage_percent,
                        "coverage_data": coverage_data,
                    }
                )
                log_coverage(
                    f"completed for {job_name} ({python_version}): {coverage_percent:.1f}%"
                )
            except Exception as e:
                log_coverage(f"failed to save for {job_name}: {e}")

        atexit.register(_stop_coverage)

    except Exception as e:
        log_coverage(f"failed to initialize: {e}")


_main()
