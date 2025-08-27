#!/usr/bin/env bash

BATCH_DIR=data/annotated_traces
MODEL_PROVIDER=openai
uv run scripts/batch_annotate/2_download_annotations.py \
    --batch-dir $BATCH_DIR \
    --model-provider $MODEL_PROVIDER \
    --api-key $LITELLM_API_KEY
