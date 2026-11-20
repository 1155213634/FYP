FROM python:3.12-slim

# Install the packages your LLM-generated code is allowed to use
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn

# Create a restricted user so LLM code doesn't run as root
RUN useradd -m sandboxuser
USER sandboxuser
WORKDIR /workspace