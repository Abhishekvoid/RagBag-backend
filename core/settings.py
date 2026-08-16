
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import timedelta

from django.core.exceptions import ImproperlyConfigured

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


ASGI_APPLICATION = "core.asgi.application"

DEBUG = os.getenv("DEBUG", "False") == "True"

# The test runner forces DEBUG=False at runtime, but settings.py is imported
# before that happens. TESTING lets the production-only "fail loudly" guards
# below stay strict in real deployments without breaking a bare `manage.py test`
# in CI, where no .env is present.
TESTING = "test" in sys.argv


def _csv(name, default=""):
    """Parse a comma-separated env var into a de-duplicated, stripped list."""
    seen = {}
    for raw in os.getenv(name, default).split(","):
        value = raw.strip()
        if value:
            seen[value] = None
    return list(seen)


def _require(value, name):
    """Production config that has no safe default. Fail loudly, never guess.

    Silently falling back to a hardcoded host is how a deployment ends up
    talking to an abandoned third-party domain.
    """
    if not value and not DEBUG and not TESTING:
        raise ImproperlyConfigured(
            f"{name} must be set when DEBUG=False. Refusing to start with an "
            f"implicit default in production."
        )
    return value


SECRET_KEY = os.environ.get("SECRET_KEY")

if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")


ALLOWED_HOSTS = _require(_csv("DJANGO_ALLOWED_HOSTS"), "DJANGO_ALLOWED_HOSTS")
if not ALLOWED_HOSTS:
    # DEBUG/TESTING only — _require() has already raised in production.
    ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]

# Application definition
INSTALLED_APPS = [

      
    'corsheaders',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    'dj_rest_auth',
    'dj_rest_auth.registration',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'rest_framework',
    'rest_framework.authtoken',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',

    'djoser',
    'storages', 
    'channels',
    'accounts',
]



MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'allauth.account.middleware.AccountMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

REDIS_URL = _require(os.getenv("REDIS_URL"), "REDIS_URL") or "redis://127.0.0.1:6379/0"

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
           "hosts": [REDIS_URL],
        },
    },
}

# DRF throttle counters and WebSocket tickets live in the cache, so it MUST be
# shared across every web process — a per-process LocMemCache silently turns
# "100 requests/hour" into "100 per gunicorn worker per deploy" and makes a
# ticket issued by one worker unusable by another.
#
# Same Redis instance as Celery/Channels (no second service), namespaced by
# KEY_PREFIX so the keyspaces cannot collide.
if DEBUG or TESTING:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "ragbag-dev",
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "KEY_PREFIX": "ragbag",
        }
    }


# Browser origins allowed to call the API, from the environment only — a new
# frontend URL is an env change, not a code change. localhost is added for local
# development ONLY; production never implicitly trusts a developer's machine,
# which matters because CORS_ALLOW_CREDENTIALS is on.
CORS_ALLOWED_ORIGINS = _require(_csv("CORS_ALLOWED_ORIGINS"), "CORS_ALLOWED_ORIGINS")
if DEBUG or TESTING:
    # dict.fromkeys preserves order while dropping duplicates.
    CORS_ALLOWED_ORIGINS = list(
        dict.fromkeys(CORS_ALLOWED_ORIGINS + ["http://localhost:3000"])
    )

CORS_ALLOW_CREDENTIALS = True

# Django must trust the HTTPS origins that POST to it (admin login, session
# endpoints) now that it sits behind a TLS-terminating proxy in production.
CSRF_TRUSTED_ORIGINS = [o for o in CORS_ALLOWED_ORIGINS if o.startswith("https://")]
CSRF_TRUSTED_ORIGINS += [
    o.strip() for o in os.getenv("CSRF_TRUSTED_ORIGINS", "").split(",") if o.strip()
]

# Render/Railway/Fly terminate TLS at a proxy and forward plain HTTP with an
# X-Forwarded-Proto header. Without this, request.is_secure() is always False,
# which breaks secure cookies and causes SSL-redirect loops.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Hardening that switches on only in production (DEBUG=False). Local dev over
# http:// is unaffected.
if not DEBUG:
    SECURE_SSL_REDIRECT = os.getenv("SECURE_SSL_REDIRECT", "True") == "True"
    # Load balancers probe over plain HTTP from inside the VPC. Without this
    # exemption SecurityMiddleware answers the probe with a 301 and the target
    # is marked unhealthy — a deploy that fails for a completely unrelated
    # reason. Only the two probe paths are exempt; everything else redirects.
    SECURE_REDIRECT_EXEMPT = [r"^ping/?$", r"^healthz/?$"]
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_SECONDS = int(os.getenv("SECURE_HSTS_SECONDS", "31536000"))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# ... (REST_FRAMEWORK, CHANNEL_LAYERS, SIMPLE_JWT, etc. are all correct)
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
   
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    # JSON only in production. The browsable HTML explorer is a development
    # convenience that otherwise ships a self-documenting UI for every endpoint
    # to the public internet.
    'DEFAULT_RENDERER_CLASSES': (
        ['rest_framework.renderers.JSONRenderer',
         'rest_framework.renderers.BrowsableAPIRenderer']
        if DEBUG else
        ['rest_framework.renderers.JSONRenderer']
    ),
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '10/hour',
        'user': '100/hour',
        'ai': '30/hour'
    }
}


SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

SITE_ID = 1

# ---------- Supabase Storage (S3-Compatible) FINAL -----------
STORAGES = {
    "default": {
        "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")

AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")

AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME")

AWS_S3_SIGNATURE_VERSION = "s3v4"
AWS_S3_ADDRESSING_STYLE = "path"

AWS_S3_FILE_OVERWRITE = False    
AWS_QUERYSTRING_AUTH = False 

MEDIA_URL = f"https://{SUPABASE_PROJECT_ID}.storage.supabase.co/storage/v1/object/public/{AWS_STORAGE_BUCKET_NAME}/"


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "studywise-documents")

if not DEBUG and not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY required in production")

# ---- Embeddings -----------------------------------------------------------
# Embedding is a HARD dependency: with no embedding service, ingestion and query
# both fail. It previously defaulted to http://localhost:8080/embed, which in
# production fails per-request inside a circuit breaker rather than at startup —
# the slowest possible way to discover a missing config. These guards move that
# failure to boot time.
EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or "tei").strip().lower()

if EMBEDDING_PROVIDER not in ("tei", "cloudflare", "openai"):
    raise ImproperlyConfigured(
        f"EMBEDDING_PROVIDER must be one of tei/cloudflare/openai, "
        f"got {EMBEDDING_PROVIDER!r}"
    )

EMBEDDING_URL = _require(os.getenv("EMBEDDING_URL"), "EMBEDDING_URL")

# A managed provider without a key is a guaranteed 401 on every request. Only
# self-hosted TEI is legitimately unauthenticated (it is not internet-facing).
if EMBEDDING_PROVIDER != "tei":
    _require(os.getenv("EMBEDDING_API_KEY"), "EMBEDDING_API_KEY")

# Pooling belongs at boot time for the same reason the URL does, only more so:
# a wrong value here does not fail a request, it returns 384 plausible floats
# from the wrong vector space. Cloudflare defaults to mean pooling while the
# index is CLS-pooled (measured cosine between the two: 0.93), so a typo like
# "CLS " or "clr" silently falling back to a provider default would degrade
# retrieval indefinitely with nothing in the logs.
EMBEDDING_POOLING = (os.getenv("EMBEDDING_POOLING") or "cls").strip().lower()

if EMBEDDING_POOLING not in ("cls", "mean"):
    raise ImproperlyConfigured(
        f"EMBEDDING_POOLING must be 'cls' or 'mean', got {EMBEDDING_POOLING!r}. "
        f"bge-small-en-v1.5 is CLS-pooled; 'mean' requires a fresh index."
    )

# Reranking is SOFT and OFF by default. Unset RERANK_URL disables it; the RAG
# pipeline falls back to vector + keyword ordering.
RERANK_URL = (os.getenv("RERANK_URL") or "").strip()
RERANK_ENABLED = bool(RERANK_URL)
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
# DATABASES = {
#     "default": {
#         "ENGINE": "django.db.backends.sqlite3",
#         "NAME": BASE_DIR / "db.sqlite3",
#     }
# }

# Tests run against an isolated in-memory SQLite DB so the suite never touches
# (or needs privileges on) the Supabase Postgres instance.
if 'test' in sys.argv:
    DATABASES['default'] = {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
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


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {                     
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
        },
        # Silence the per-request HTTP log wall from httpx/httpcore (Groq, TEI,
        # HuggingFace hub) and the low-level HTTP/2 framing chatter.
        "httpx": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "httpcore": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "hpack": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
}

AUTH_USER_MODEL = 'accounts.CustomUserModel'

# Django applies NO password rules unless this list exists — an absent setting
# is an empty list, and validate_password() then accepts literally anything,
# including "1". Djoser's registration serializer calls validate_password(), so
# populating this list is what actually enforces a policy at signup.
#
# min_length 10 rather than Django's default 8: a modest bump that stays well
# inside NIST 800-63B guidance and does not push users toward writing passwords
# down. No composition rules (no "must contain a symbol") — those measurably
# harm usability without improving entropy.
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 10},
    },
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

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