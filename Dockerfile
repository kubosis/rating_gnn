ARG BASE_IMAGE=python:3.11
FROM ${BASE_IMAGE} as base

WORKDIR /nera

ENV PYTHONUNBUFFERED TRUE

# Copy only dependencies first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir wheel \
    && pip install torch==2.1.2 \
    && pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html \
    && pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cpu.html \
    && pip install torch-geometric==2.4.0 \
    && pip install torch-geometric-temporal==0.52.0 \
    && pip install -r requirements.txt \
    && pip list

COPY . /nera

EXPOSE 9999

CMD ["python", "src/evaluation.py"]
