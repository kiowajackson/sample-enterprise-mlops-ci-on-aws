FROM public.ecr.aws/docker/library/python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Both example models use this same resolved runtime.
COPY mlops-models/src/mlops_models/models/usecase1/requirements.lock.txt /tmp/model-requirements.lock.txt
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/model-requirements.lock.txt \
    && python -m pip check \
    && rm /tmp/model-requirements.lock.txt \
    && useradd --uid 10001 --create-home mlops

COPY mlops-models/src/mlops_models /opt/program/mlops_models
COPY mlops-models/mlops_skeleton /opt/program/mlops_skeleton

ENV PYTHONPATH=/opt/program \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/mlops \
    TMPDIR=/home/mlops \
    TMP=/home/mlops \
    TEMP=/home/mlops \
    MPLCONFIGDIR=/home/mlops/matplotlib
WORKDIR /opt/program
USER 10001:10001
ENTRYPOINT ["python", "-m", "mlops_models.entrypoints"]
