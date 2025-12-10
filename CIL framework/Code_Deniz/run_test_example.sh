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

# Example 2: Test ORCO model (exp3 - partial unfreeze)
echo "Testing ORCO exp3 (partial unfreeze) with 1-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/ORCO_exp3_partial_unfreeze_better_best.pt \
    --strategy ORCO \
    --k_shot 1 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/ORCO_exp3_1shot

# Example 3: Test ORCO model (exp3 - partial unfreeze)
echo "Testing CONCM exp3 (partial unfreeze) with 1-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/CONCM_exp3_partial_unfreeze_better_best.pt \
    --strategy CONCM \
    --k_shot 1 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/CONCM_exp3_1shot


# Example 4: Test ORCO model (exp3 - partial unfreeze)
echo "Testing ORCO exp3 (partial unfreeze) with 5-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/ORCO_exp3_partial_unfreeze_better_best.pt \
    --strategy ORCO \
    --k_shot 5 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/ORCO_exp3_5shot

# Example 5: Test CONCM model (exp3 - partial unfreeze)
echo "Testing CONCM exp3 (partial unfreeze) with 5-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/CONCM_exp3_partial_unfreeze_better_best.pt \
    --strategy CONCM \
    --k_shot 5 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/CONCM_exp3_5shot

# Example 6: Test ORCO model (exp3 - partial unfreeze)
echo "Testing ORCO exp3 (partial unfreeze) with 10-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/ORCO_exp3_partial_unfreeze_better_best.pt \
    --strategy ORCO \
    --k_shot 10 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/ORCO_exp3_10shot

# Example 7: Test CONCM model (exp3 - partial unfreeze)
echo "Testing CONCM exp3 (partial unfreeze) with 10-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/CONCM_exp3_partial_unfreeze_better_best.pt \
    --strategy CONCM \
    --k_shot 10 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/CONCM_exp3_10shot

# Example 8: Test CONCM model (exp3 - partial unfreeze)
echo "Testing CONCM_FULL exp3 (partial unfreeze) with 1-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/CONCM_FULL_exp3_partial_unfreeze_better_best.pt \
    --strategy CONCM_FULL \
    --k_shot 1 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/CONCM_FULL_exp3_1shot

# Example 9: Test CONCM_FULL model (exp3 - partial unfreeze)
echo "Testing CONCM_FULL exp3 (partial unfreeze) with 5-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/CONCM_FULL_exp3_partial_unfreeze_better_best.pt \
    --strategy CONCM_FULL \
    --k_shot 5 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/CONCM_FULL_exp3_5shot

# Example 10: Test CONCM_FULL model (exp3 - partial unfreeze)
echo "Testing CONCM_FULL exp3 (partial unfreeze) with 10-shot..."
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/CONCM_FULL_exp3_partial_unfreeze_better_best.pt \
    --strategy CONCM_FULL \
    --k_shot 10 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv \
    --output_dir ./test_results/CONCM_FULL_exp3_10shot