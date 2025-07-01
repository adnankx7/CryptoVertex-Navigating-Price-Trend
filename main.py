import subprocess
import sys
import pathlib
import time
from datetime import datetime, timedelta
from src.logger import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [MAIN_ORCHESTRATOR] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent

def run_script(script_path: str, project_root: pathlib.Path) -> bool:
    full_script_path = project_root / script_path
    logging.info(f"Attempting to run script: {full_script_path}")

    if not full_script_path.exists():
        logging.info(f"Script not found at {full_script_path}")
        return False

    logging.info(f"Starting execution of: {script_path}")
    try:
        process = subprocess.run(
            [sys.executable, str(full_script_path)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False
        )

        if process.stdout:
            logging.info(f"Output from {script_path}:\n{process.stdout.strip()}")
        if process.stderr:
            logging.info(f"Error output from {script_path}:\n{process.stderr.strip()}")

        if process.returncode == 0:
            logging.info(f"Successfully finished execution of: {script_path}")
            return True
        else:
            logging.info(f"Script {script_path} failed with return code {process.returncode}.")
            return False
    except Exception as e:
        logging.info(f"Unexpected error while running {script_path}: {e}")
        return False

def run_pipeline():
    project_root = get_project_root()
    logging.info("🚀 Running pipeline...")

    scripts_to_run = [
        pathlib.Path("src") / "scraper" / "scraper.py", 
        pathlib.Path("src") / "components" / "data_transformation.py",
        pathlib.Path("src") / "components" / "data_trainer.py",
        pathlib.Path("src") / "pipeline" / "predict_pipeline.py"
    ]

    for script_rel_path in scripts_to_run:
        if not run_script(str(script_rel_path), project_root):
            logging.info(f"❌ Pipeline stopped due to failure in {script_rel_path}.")
            return

    logging.info("✅ Pipeline execution completed.")

def run_scheduler():
    logging.info("🔄 Waiting for daily run at 05:00:10 AM local time...")

    while True:
        now = datetime.now()
        target_time = now.replace(hour=5, minute=0, second=10, microsecond=0)

        # If current time already past today’s 5:00:10, schedule for next day
        if now >= target_time:
            target_time += timedelta(days=1)

        wait_seconds = (target_time - now).total_seconds()
        logging.info(f"⏳ Sleeping for {int(wait_seconds)} seconds until next 05:00:10 AM...")

        try:
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            logging.info("Scheduler interrupted by user. Exiting...")
            break

        logging.info(f"🕔 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local time — running pipeline...")
        run_pipeline()

        # Sleep 24 hours before next run
        try:
            time.sleep(24 * 60 * 60)
        except KeyboardInterrupt:
            logging.info("Scheduler interrupted by user. Exiting...")
            break

if __name__ == "__main__":
    run_scheduler()