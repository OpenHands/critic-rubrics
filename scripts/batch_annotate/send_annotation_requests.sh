#!/usr/bin/env bash

TRACE_DIR=data/compressed_traces
OUTPUT_DIR=data/annotated_traces
LIMIT=5

MODEL=litellm_proxy/openai/o3-2025-04-16
MODEL_PROVIDER=openai

uv run scripts/batch_annotate/1_send_annotation_requests.py \
    --trace-dir $TRACE_DIR \
    --output-dir $OUTPUT_DIR \
    --limit $LIMIT \
    --model $MODEL \
    --model-provider $MODEL_PROVIDER \
    --api-key $LITELLM_API_KEY
