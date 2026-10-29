# PandoRSI lab image (Ring 2): runs replication experiments inside a
# disposable container. The lab gate detects /.dockerenv automatically.
#
# Build:  docker build -t pando-lab .
# Run:    docker run --rm pando-lab          (runs the gated lab demo)
#         docker run --rm pando-lab python -m pando.doctor --json
#
# NOTE: written for environments with Docker; not yet build-tested by the
# authors on this repo (no Docker in the current dev environment at the
# time of writing). Issue reports welcome.

FROM python:3.12-slim

WORKDIR /pando
COPY pando/ pando/
COPY assets/ assets/
RUN pip install --no-cache-dir pynacl zeroconf

# The lab gate detects the container via /.dockerenv; the env var is a
# belt-and-braces operator attestation.
ENV PANDOSI_CONTAINER=1

CMD ["python", "-m", "pando.lab"]
