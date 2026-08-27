FROM python:3.11.10-slim-bookworm

WORKDIR /workspace/backend
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "psycopg[binary]==3.3.4" \
    "python-dotenv==1.2.3"

COPY backend/db ./db

USER 65532:65532
ENTRYPOINT ["python", "db/provision.py"]
CMD ["all"]
