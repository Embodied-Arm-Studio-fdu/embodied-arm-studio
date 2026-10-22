FROM ghcr.io/embodied-arm-studio-fdu/embodied-arm-studio-a7-worker@sha256:41dccb9d880bddc0fba2b7df60a5d5695a38290057f76f622c6ec00a41f84e89
COPY simulation/scripts/push_diagnostic_training.py /app/simulation/scripts/push_diagnostic_training.py
