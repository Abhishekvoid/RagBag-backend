# in core/celery.py

import os
from celery import Celery


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')


app = Celery('core')


app.config_from_object('core.celeryconfig')

app.autodiscover_tasks()