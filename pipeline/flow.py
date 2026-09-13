import schedule
import time
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
def run_step(script_path: str, step_name: str) -> bool:
    logging.info(f"Starting step: {step_name}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        logging.info(f"Step succeeded: {step_name}")
        logging.info(result.stdout.strip())
        return True
    else:
        logging.error(f"Step failed: {step_name}")
        logging.error(result.stderr.strip())
        return False


def run_pipeline():
    logging.info("=== Pipeline run started ===")

    generate_ok = run_step("pipeline/generate_data.py", "generate_data")
    if not generate_ok:
        logging.error("Pipeline stopped: generate_data failed")
        return

    transform_ok = run_step("pipeline/transform.py", "transform")
    if not transform_ok:
        logging.error("Pipeline stopped: transform failed")
        return

    logging.info("=== Pipeline run completed successfully ===")
schedule.every().day.at("09:00").do(run_pipeline)

if __name__ == "__main__":
    logging.info("Scheduler started. Waiting for scheduled runs (daily at 09:00)...")
    run_pipeline()  # run once immediately so you can see it work right away

    while True:
        schedule.run_pending()
        time.sleep(60)