import subprocess
import sys
import pathlib
from src.logger import logging
import schedule
import time
from datetime import datetime, timezone, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [MAIN_ORCHESTRATOR] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

UTC_PLUS_5 = timezone(timedelta(hours=5))

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
    logging.info("Running pipeline...")

    scripts_to_run = [
        pathlib.Path("src") / "scraper.py",
        pathlib.Path("src") / "components" / "data_transformation.py",
        pathlib.Path("src") / "components" / "data_trainer.py",
        pathlib.Path("src") / "pipeline" / "predict_pipeline.py"
    ]

    for script_rel_path in scripts_to_run:
        if not run_script(str(script_rel_path), project_root):
            logging.info(f"Pipeline stopped due to failure in {script_rel_path}.")
            return

    logging.info("Pipeline execution completed.")

def get_utc_now_plus_5_time():
    return datetime.now(timezone.utc).astimezone(UTC_PLUS_5).strftime("%H:%M")

def run_scheduler():

    RUN_TIME_UTC_PLUS_5 = "00:00"

    schedule.every().day.at(RUN_TIME_UTC_PLUS_5).do(run_pipeline)

    logging.info(f"Scheduled daily pipeline run at {RUN_TIME_UTC_PLUS_5} UTC+5")

    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    run_scheduler()
