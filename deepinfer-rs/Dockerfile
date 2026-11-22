# DeepInfer-rs serving image.
#
# Build:   docker build -t deepinfer-server .
# Run:     docker run --init --gpus all -p 8080:8080 \
#            -v /path/to/DeepSeek-V4.1-Flash-1017:/model:ro \
#            deepinfer-server --model /model
#
# --init propagates SIGTERM to the server for a clean drain (the Rust
# binary relies on the init system for signal delivery; without it,
# `docker stop` waits the 10s grace period and SIGKILLs).
#
# CPU-only variant: drop --gpus all.

FROM rust:1.100-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p deepinfer-server --bin deepinfer-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/deepinfer-server /usr/local/bin/deepinfer-server
# Non-root runtime user.
RUN useradd --system --create-home deepinfer
USER deepinfer
EXPOSE 8080
ENV DEEPINFER_SERVER_HOST=0.0.0.0 \
    DEEPINFER_SERVER_PORT=8080
ENTRYPOINT ["deepinfer-server"]
CMD ["--tiny"]
