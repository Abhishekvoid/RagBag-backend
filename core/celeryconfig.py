import os

broker_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
result_backend = broker_url


# Standard Celery settings.
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True