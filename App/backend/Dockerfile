FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    TUNISCOPE_MODEL_DIR=/srv/App/models/current
WORKDIR /srv
COPY App/backend/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY scripts/normalize_text.py /srv/scripts/normalize_text.py
COPY App/backend/service /srv/App/backend/service
COPY App/models/current /srv/App/models/current
ARG REQUIRE_MODELS=true
# Fail the build for missing, incompatible or corrupted models by default.
RUN TUNISCOPE_REQUIRE_MODELS=${REQUIRE_MODELS} python -m App.backend.service.check_bundle
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && chmod -R a-w /srv/App/models
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "App.backend.service.main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api", "--no-access-log"]
