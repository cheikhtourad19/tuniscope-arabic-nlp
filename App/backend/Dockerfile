FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    TUNISCOPE_MODEL_DIR=/srv/App/models/current
WORKDIR /srv
COPY App/backend/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.lock
COPY App/models/current /srv/App/models/current
COPY --chown=10001:10001 App/models/neural /srv/App/models/neural
COPY scripts/normalize_text.py scripts/qwen_prompting.py scripts/fetch_qwen3.py /srv/scripts/
COPY configs/qwen_prompting_v2.json configs/rag_e5_small_v1.json /srv/configs/
COPY rag/ /srv/rag/
COPY docs/ /srv/docs/
COPY SOURCE_REGISTER.md /srv/SOURCE_REGISTER.md
COPY reports/rag_validation_v1_2026-11-09/retrieval.json /srv/reports/rag_validation_v1_2026-11-09/retrieval.json
COPY App/backend/service /srv/App/backend/service
ARG REQUIRE_MODELS=true
ARG REQUIRE_NEURAL=true
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
USER 10001:10001
# Verify the same access and model loading as the non-root API process.
RUN TUNISCOPE_REQUIRE_MODELS=${REQUIRE_MODELS} TUNISCOPE_REQUIRE_NEURAL=${REQUIRE_NEURAL} python -m App.backend.service.check_bundle
EXPOSE 8000
CMD ["uvicorn", "App.backend.service.main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api", "--no-access-log"]
