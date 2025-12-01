# Use the official Rust image as a builder
FROM rust:1.78 as builder

# Create a new empty shell project
WORKDIR /usr/src/skyscope-core-rs
COPY . .

# Build for release
RUN cargo build --release

# The final, smaller image
FROM debian:buster-slim
RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

# Copy the build artifact from the builder stage
COPY --from=builder /usr/src/skyscope-core-rs/target/release/skyscope-core-rs .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the executable
CMD ["./skyscope-core-rs"]
