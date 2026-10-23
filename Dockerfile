FROM python:3.11-slim
WORKDIR /app
COPY requirements-portable.txt .
RUN pip install --no-cache-dir -r requirements-portable.txt
COPY wallex_lab ./wallex_lab
COPY config ./config
COPY docs/fee-evidence ./docs/fee-evidence
ENV PYTHONUNBUFFERED=1 LLM_PROVIDER=openai_compatible LANGSMITH_TRACING=false
ENTRYPOINT ["python", "-m", "wallex_lab.portable"]
CMD ["probe"]
