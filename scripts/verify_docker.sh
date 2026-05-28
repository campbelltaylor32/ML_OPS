#!/usr/bin/env bash
# scripts/verify_docker.sh
# Canonical end-to-end verification using Docker (matches CI exactly).
# Run this before pushing to confirm everything works inside the image.
#
# Steps:
#   1. Build pipeline artifacts via dvc repro
#   2. Run all guard tests (pytest)
#   3. Verify serving stack starts and responds
set -e

echo "=== Step 1: Build pipeline artifacts (dvc repro inside container) ==="
docker compose run --rm pipeline

echo ""
echo "=== Step 2: Run guard tests inside container ==="
docker compose run --rm test

echo ""
echo "=== Step 3: Smoke-test deployable apps ==="
bash scripts/smoke_apps.sh

echo ""
echo "All verification steps passed."
echo "Start the full serving stack with: docker compose up"
