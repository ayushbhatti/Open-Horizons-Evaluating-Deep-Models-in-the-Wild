#!/bin/bash
# Example script to run testing on trained models with fixed samples

# Make sure you're in the correct directory
cd "$(dirname "$0")"

# Generate fixed samples if they don't exist
if [ ! -f "fixed_test_samples.csv" ]; then
    echo "Generating fixed test samples..."
    python generate_test_samples.py
    echo ""
fi

# Example 0: Test BASELINE model (trained from scratch only on base classes, no incremental learning)
echo "Testing BASELINE exp2 (scratch) - no incremental learning..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/BASELINE_exp2_scratch_fulltrain_best.pt \
    --strategy BASELINE \
    --k_shot 5 \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/BASELINE_exp2

# Example 1: Test ORCO model (exp1 - pretrained frozen)
echo "Testing ORCO exp1 (pretrained frozen) with 5-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/ORCO_exp1_pretrained_frozen_best.pt \
    --strategy ORCO \
    --k_shot 5 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/ORCO_exp1


# Example 4: Test CONCM model (exp1 - pretrained frozen)
echo "Testing CONCM exp1 (pretrained frozen) with 5-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/CONCM_exp1_pretrained_frozen_best.pt \
    --strategy CONCM \
    --k_shot 5 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/CONCM_exp1

# Example 5: Test ORCO model (exp2 - scratch full train) - NO pretrained flag
echo "Testing ORCO exp2 (scratch full train) with 5-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/ORCO_exp2_scratch_fulltrain_best.pt \
    --strategy ORCO \
    --k_shot 5 \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/ORCO_exp2

# Example 6: Test ORCO model (exp3 - partial unfreeze)
echo "Testing ORCO exp3 (partial unfreeze) with 5-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/ORCO_exp3_partial_unfreeze_better_best.pt \
    --strategy ORCO \
    --k_shot 5 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/ORCO_exp3

echo ""
echo "All tests completed! Check the test_results/ directory for outputs."
echo "All experiments used the same fixed test samples for consistent comparison."

