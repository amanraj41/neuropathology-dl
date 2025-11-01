#!/bin/bash
# Post-training evaluation script
# Run this after Stage 2 training completes

echo "=========================================="
echo "POST-TRAINING EVALUATION"
echo "=========================================="
echo ""

# Check if training is complete
if ps -p 49355 > /dev/null 2>&1; then
    echo "⚠️  Training still running (PID 49355)"
    echo "   Wait for training to complete, then run this script again."
    exit 1
fi

echo "✓ Training completed"
echo ""

# Show final validation accuracies
echo "=== Final Validation Accuracies ==="
python3 << PYEOF
import pandas as pd
df1 = pd.read_csv('logs/train/history_stage1.csv')
df2 = pd.read_csv('logs/train/history_stage2.csv')

best1_idx = df1['val_accuracy'].idxmax()
best2_idx = df2['val_accuracy'].idxmax()

print(f"Stage 1 (frozen):   Epoch {int(df1.loc[best1_idx, 'epoch']):2d} -> {df1.loc[best1_idx, 'val_accuracy']:.6f} val_accuracy")
print(f"Stage 2 (finetuned): Epoch {int(df2.loc[best2_idx, 'epoch']):2d} -> {df2.loc[best2_idx, 'val_accuracy']:.6f} val_accuracy")
print(f"\nImprovement: +{((df2.loc[best2_idx, 'val_accuracy'] - df1.loc[best1_idx, 'val_accuracy']) * 100):.2f} percentage points")
PYEOF

echo ""
echo "=== Re-evaluating Test Accuracies ==="
echo "This will compute honest test set performance for all models..."
echo ""

# Re-evaluate all models
python scripts/evaluate_models.py \
    --data_dir ./data/brain_mri_17 \
    --batch_size 16 \
    --models models/best_model.keras models/best_model_finetuned.keras models/final_model.keras

echo ""
echo "=== Summary ==="
echo "Updated metrics files:"
echo "  - models/metrics_best_model.json (Stage 1 frozen)"
echo "  - models/metrics_best_model_finetuned.json (Stage 2 finetuned)"
echo "  - models/metrics_final_model.json (final snapshot)"
echo ""
echo "✓ App will now display correct test accuracies for each model!"
echo ""
echo "View in app:"
echo "  streamlit run app.py"
echo "  Then use the dropdown to select different models and see their test accuracies."
