#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# Prefix with sortable timestamp so plain `ls logs` shows oldest at top, newest at bottom.
timestamp="$(date +%Y%m%d_%H%M%S)"

sbatch \
  --output="logs/${timestamp}_%x_%j.out" \
  --error="logs/${timestamp}_%x_%j.err" \
  "$@" \
  slurm/main.slurm
