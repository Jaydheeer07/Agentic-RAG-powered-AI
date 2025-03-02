FROM postgres:15

# Install build dependencies and pgvector
RUN apt-get update \
    && apt-get install -y \
        postgresql-server-dev-15 \
        git \
        build-essential \
    && cd /tmp \
    && git clone --branch v0.4.4 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && apt-get remove -y postgresql-server-dev-15 git build-essential \
    && apt-get autoremove -y \
    && rm -rf /tmp/pgvector \
    && rm -rf /var/lib/apt/lists/*
