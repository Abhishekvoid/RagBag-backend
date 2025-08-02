from celery import shared_task
import time

@shared_task
def simple_test_task():
    """A simple task that just prints a message to the Celery worker console."""
    print("Starting the test task...")
    time.sleep(5) 
    print("SUCCESS: The Celery task has finished running!")
    return "Task completed successfully."