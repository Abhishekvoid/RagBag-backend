# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from datetime import timedelta

# load_dotenv()

# BASE_DIR = Path(__file__).resolve().parent.parent



# DEBUG = os.getenv("DEBUG", "False") == "True"

# SECRET_KEY = os.getenv('SECRET_KEY')
# if not SECRET_KEY:
#     raise ValueError("SECRET_KEY must be set in .env file")


# ALLOWED_HOSTS = [
#     host.strip() 
#     for host in os.getenv('DJANGO_ALLOWED_HOSTS', '').split(',') 
#     if host.strip()
# ]




# # Application definition

# INSTALLED_APPS = [
#     'django.contrib.admin',
#     'django.contrib.auth',
#     'django.contrib.contenttypes',
#     'django.contrib.sessions',
#     'django.contrib.messages',
#     'django.contrib.staticfiles',
#     'dj_rest_auth',
#     'dj_rest_auth.registration',
#     'allauth',
#     'allauth.account',
#     'allauth.socialaccount',
#     'allauth.socialaccount.providers.google',
#     'rest_framework',
#     'rest_framework.authtoken',
#     'rest_framework_simplejwt',
#     'corsheaders',
#     'djoser',
#     'channels',
#     'storages',
#     'accounts',
# ]

# MIDDLEWARE = [
#     'django.middleware.security.SecurityMiddleware',
#     'whitenoise.middleware.WhiteNoiseMiddleware',

#     'corsheaders.middleware.CorsMiddleware',
#     'allauth.account.middleware.AccountMiddleware',
#     'django.contrib.sessions.middleware.SessionMiddleware',
#     'django.middleware.common.CommonMiddleware',
#     'django.middleware.csrf.CsrfViewMiddleware',
#     'django.contrib.auth.middleware.AuthenticationMiddleware',
#     'django.contrib.messages.middleware.MessageMiddleware',
#     'django.middleware.clickjacking.XFrameOptionsMiddleware',
# ]

# ROOT_URLCONF = 'core.urls'



# CORS_ALLOWED_ORIGINS = [
#     origin.strip() 
#     for origin in os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
#     if origin.strip()
# ]
# CORS_ALLOW_CREDENTIALS = True

# REST_FRAMEWORK = {
#     'DEFAULT_AUTHENTICATION_CLASSES': (
#         'rest_framework_simplejwt.authentication.JWTAuthentication',
#     ),
#     'DEFAULT_RENDERER_CLASSES': (
#         'rest_framework.renderers.JSONRenderer',
#         'rest_framework.renderers.BrowsableAPIRenderer',
#     ),
# }

# REST_FRAMEWORK['DEFAULT_THROTTLE_CLASSES'] = [
#     'rest_framework.throttling.AnonRateThrottle',
#     'rest_framework.throttling.UserRateThrottle'
# ]
# REST_FRAMEWORK['DEFAULT_THROTTLE_RATES'] = {
#     'anon': '10/hour',
#     'user': '100/hour'
# }

# CHANNEL_LAYERS = {
#     "default": {
#         "BACKEND": "channels_redis.core.RedisChannelLayer",
#         "CONFIG": {
#             "hosts": [("127.0.0.1", 6379)],
#         },
#     },
# }
# SIMPLE_JWT = {
#     'ACCESS_TOKEN_LIFETIME': timedelta(
#         minutes=int(os.getenv('JWT_ACCESS_TOKEN_LIFETIME_MINUTES', 15))
#     ),
#     'REFRESH_TOKEN_LIFETIME': timedelta(
#         days=int(os.getenv('JWT_REFRESH_TOKEN_LIFETIME_DAYS', 7))
#     ),
#     'ROTATE_REFRESH_TOKENS': True,
#     'BLACKLIST_AFTER_ROTATION': True,
#     'AUTH_HEADER_TYPES': ('Bearer',),
#     'ALGORITHM': 'HS256',
# }
# SITE_ID = 1

# ACCOUNT_LOGIN_METHODS = {'email', 'username'}
# ACCOUNT_SIGNUP_FIELDS = ['email*', 'username*', 'password1*', 'password2*']
 
# CELERY_BROKER_URL = 'redis://localhost:6379/0'
# CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
# CELERY_ACCEPT_CONTENT = ['json']
# CELERY_TASK_SERIALIZER = 'json'
# CELERY_RESULT_SERIALIZER = 'json'
# CELERY_TIMEZONE = 'UTC'

# # ---------- S3boto3storage -----------


# DEFAULT_FILE_STORAGE = 'storages.backends.s3_boto3.S3Boto3Storage'


# SUPABASE_PROJECT_ID = os.getenv('SUPABASE_PROJECT_ID')
# SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET')


# AWS_ACCESS_KEY_ID = os.getenv('SUPABASE_ACCESS_KEY')
# AWS_SECRET_ACCESS_KEY = os.getenv('SUPABASE_SECRET_KEY')
# AWS_STORAGE_BUCKET_NAME = SUPABASE_BUCKET
# AWS_S3_REGION_NAME = os.getenv('SUPABASE_REGION')


# AWS_S3_ENDPOINT_URL = f"https://{SUPABASE_PROJECT_ID}.supabase.co/storage/v1"


# AWS_S3_CUSTOM_DOMAIN = f"{SUPABASE_PROJECT_ID}.supabase.co/storage/v1/object/public/{SUPABASE_BUCKET}"

# MEDIA_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/"

# AWS_S3_SIGNATURE_VERSION = 's3v4'

# AWS_S3_FILE_OVERWRITE = False

# TEMPLATES = [
#     {
#         'BACKEND': 'django.template.backends.django.DjangoTemplates',
#         'DIRS': [],
#         'APP_DIRS': True,
#         'OPTIONS': {
#             'context_processors': [
#                 'django.template.context_processors.debug',
#                 'django.template.context_processors.request',
#                 'django.contrib.auth.context_processors.auth',
#                 'django.contrib.messages.context_processors.messages',
#             ],
#         },
#     },
# ]

# WSGI_APPLICATION = 'core.wsgi.application'


# SECURE_BROWSER_XSS_FILTER = True
# SECURE_CONTENT_TYPE_NOSNIFF = True
# SESSION_COOKIE_SECURE = not DEBUG
# CSRF_COOKIE_SECURE = not DEBUG
# X_FRAME_OPTIONS = "DENY"

# # Database
# # https://docs.djangoproject.com/en/5.2/ref/settings/#databases

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql_psycopg2',
#         'HOST': os.getenv('SUPABASE_DB_HOST'),
#         'NAME': os.getenv('SUPABASE_DB_NAME'),
#         'USER': os.getenv('SUPABASE_DB_USER'),
#         'PORT': os.getenv('SUPABASE_DB_PORT', '5432'),
#         'PASSWORD': os.getenv('SUPABASE_DB_PASSWORD'),
#         'OPTIONS': {
#             'sslmode': 'require',
#         },
#     }
# }
# # ✅ THIS IS THE CORRECTED LOGGING CONFIGURATION
# LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'formatters': {
#         'verbose': {
#             'format': '{levelname} {asctime} {module} : {message}',
#             'style': '{',
#         },
#     },
#     'handlers': {
#         'console': {
#             'level': 'INFO',
#             'class': 'logging.StreamHandler',
#             'formatter': 'verbose',
#         },
#         'file': {
#             'level': 'INFO',
#             'class': 'logging.FileHandler',
#             'filename': 'django.log',
#             'formatter': 'verbose',
#         },
#     },
#     'root': {
#         'handlers': ['console', 'file'], # Now sends to both terminal and file
#         'level': 'INFO',
#     },
# }
# # Password validation
# # https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

# AUTH_USER_MODEL = 'accounts.CustomUserModel'


# AUTH_PASSWORD_VALIDATORS = [
#     {
#         'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
#     },
#     {
#         'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
#     },
#     {
#         'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
#     },
#     {
#         'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
#     },
# ]


# # Internationalization
# # https://docs.djangoproject.com/en/5.2/topics/i18n/

# LANGUAGE_CODE = 'en-us'

# TIME_ZONE = 'UTC'

# USE_I18N = True

# USE_TZ = True


# # Static files (CSS, JavaScript, Images)
# # https://docs.djangoproject.com/en/5.2/howto/static-files/

# STATIC_URL = "/static/"
# STATIC_ROOT = BASE_DIR / "staticfiles"

# MEDIA_ROOT = BASE_DIR / "media"


# LOGIN_REDIRECT_URL = '/'



# DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'  

import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


ASGI_APPLICATION = "core.asgi.application"

DEBUG = os.getenv("DEBUG", "False") == "True"

SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in .env file")

ALLOWED_HOSTS = [
    host.strip() 
    for host in os.getenv('DJANGO_ALLOWED_HOSTS', '127.0.0.1,localhost').split(',') 
    if host.strip()
]

# Application definition
INSTALLED_APPS = [

      

    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'dj_rest_auth',
    'dj_rest_auth.registration',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    'rest_framework',
    'rest_framework.authtoken',
    'rest_framework_simplejwt',
    'corsheaders',
    'djoser',
    'storages', # The django-storages app
    'channels',
    'accounts',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'allauth.account.middleware.AccountMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("127.0.0.1", 6379)],
        },
    },
}


CORS_ALLOWED_ORIGINS = [
    origin.strip() 
    for origin in os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = True

# ... (REST_FRAMEWORK, CHANNEL_LAYERS, SIMPLE_JWT, etc. are all correct)
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
   
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ),
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '10/hour',
        'user': '100/hour'
    }
}
# ---------- Supabase Storage (S3-Compatible) FINAL -----------
DEFAULT_FILE_STORAGE = 'storages.backends.s3_boto3.S3Boto3Storage'

SUPABASE_PROJECT_ID = os.getenv('SUPABASE_PROJECT_ID')
SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET')

AWS_ACCESS_KEY_ID = os.getenv('SUPABASE_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('SUPABASE_SECRET_KEY')
AWS_STORAGE_BUCKET_NAME = SUPABASE_BUCKET
AWS_S3_REGION_NAME = os.getenv('SUPABASE_REGION')

AWS_S3_ENDPOINT_URL = f"https://{SUPABASE_PROJECT_ID}.supabase.co/storage/v1"
AWS_S3_CUSTOM_DOMAIN = f"{SUPABASE_PROJECT_ID}.supabase.co/storage/v1/object/public/{SUPABASE_BUCKET}"
MEDIA_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/"

AWS_S3_SIGNATURE_VERSION = 's3v4'
AWS_S3_FILE_OVERWRITE = False

# ... (TEMPLATES, WSGI_APPLICATION, SECURITY SETTINGS are correct) ...

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'HOST': os.getenv('SUPABASE_DB_HOST'),
        'NAME': os.getenv('SUPABASE_DB_NAME'),
        'USER': os.getenv('SUPABASE_DB_USER'),
        'PORT': os.getenv('SUPABASE_DB_PORT', '5432'),
        'PASSWORD': os.getenv('SUPABASE_DB_PASSWORD'),
        'OPTIONS': { 'sslmode': 'require' },
    }
}

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# Corrected Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} : {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'django.log',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

AUTH_USER_MODEL = 'accounts.CustomUserModel'
# ... (AUTH_PASSWORD_VALIDATORS are correct) ...

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"



LOGIN_REDIRECT_URL = '/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


ACCOUNT_EMAIL_VERIFICATION = 'none'

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'