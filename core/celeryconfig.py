broker_url = 'redis://localhost:6379/0'

# Set the result backend to use Redis.
result_backend = 'redis://localhost:6379/0'

# Standard Celery settings.
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True