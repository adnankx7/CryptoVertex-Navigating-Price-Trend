from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import logging
import asyncio

# Import your task functions
# Note: Adjust import paths based on your actual structure
from src.scraper.scraper import run_scrape
from src.components.data_trainer import run_training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

async def wrapped_scrape():
    """Wrapper to run synchronous scrape function asynchronously"""
    logger.info("Starting scheduled scrape job...")
    try:
        # Run in a separate thread to avoid blocking the event loop
        await asyncio.to_thread(run_scrape)
        logger.info("Scheduled scrape job completed.")
    except Exception as e:
        logger.error(f"Error in scrape job: {e}")

async def wrapped_training():
    """Wrapper to run synchronous training function asynchronously"""
    logger.info("Starting scheduled training job...")
    try:
        # Run in a separate thread to avoid blocking the event loop
        await asyncio.to_thread(run_training)
        logger.info("Scheduled training job completed.")
    except Exception as e:
        logger.error(f"Error in training job: {e}")

def start_scheduler():
    if not scheduler.running:
        # 1. Scraping: Daily at 4:00 AM
        scheduler.add_job(
            wrapped_scrape,
            CronTrigger(hour=4, minute=0),
            id='daily_scrape',
            replace_existing=True,
            name='Daily Scrape at 4 AM'
        )

        # 2. Training: Every 2 weeks
        scheduler.add_job(
            wrapped_training,
            IntervalTrigger(weeks=2),
            id='biweekly_training',
            replace_existing=True,
            name='Bi-weekly Training'
        )

        # 3. Startup logic: Run both immediately if needed
        # We add them as one-off jobs to run "now"
        scheduler.add_job(
            wrapped_scrape,
            'date',
            run_date=datetime.now() + timedelta(seconds=10), # Small delay to let server start
            id='startup_scrape',
            name='Startup Scrape'
        )
        
        scheduler.add_job(
            wrapped_training,
            'date',
            run_date=datetime.now() + timedelta(seconds=20), # Small delay
            id='startup_training',
            name='Startup Training'
        )

        scheduler.start()
        logger.info("Scheduler started with daily scrape (4am) and bi-weekly training.")
