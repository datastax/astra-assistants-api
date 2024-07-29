# Use a base image that has the correct Python version
FROM python:3.10.12-slim

# Install build dependencies required for packages with C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry
# Avoids the need to add Poetry to PATH
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Set the working directory
WORKDIR /usr/src/app

# Copy only the files necessary for installing dependencies
COPY pyproject.toml poetry.lock* ./

# Disable the creation of virtual environments by Poetry
# as the Docker container itself provides isolation.
RUN poetry config virtualenvs.create false

# Install the dependencies specified in `pyproject.toml` and `poetry.lock`.
# Use `--no-root` to avoid installing the main package and `--no-dev` to exclude development dependencies.
RUN poetry install --no-root --no-dev

# Copy the rest of your application files into the container
COPY ./openapi_server ./openapi_server
COPY ./openapi_server_v2 ./openapi_server_v2
COPY ./impl ./impl
COPY ./tests ./tests

ENV PROMETHEUS_MULTIPROC_DIR=/tmp

# Specify the command to run your application
CMD ["poetry", "run", "uvicorn", "impl.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "600"]
