#!/bin/bash
# Quick training status check

PID=81478
if ps -p $PID > /dev/null 2>&1; then
    ELAPSED=$(ps -p $PID -o etime --no-headers | tr -d ' ')
    echo "✓ Training RUNNING (PID: $PID, Elapsed: $ELAPSED)"
    echo ""
    echo "Latest epochs:"
    tail -8 logs/train/run-20251026-141823.log | grep -E "Epoch [0-9]+/60"
    echo ""
    echo "To monitor live: tail -f logs/train/run-20251026-141823.log"
else
    echo "✗ Training COMPLETED or stopped"
    echo ""
    echo "Run post-training evaluation:"
    echo "  bash scripts/post_training_evaluation.sh"
fi
