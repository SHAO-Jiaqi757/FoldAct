"""
Per-Turn Training Example

This example shows how to use per-turn context saving for correct multi-turn training.
"""

import torch
from verl import DataProto
from verl.utils.per_turn_training import PerTurnContextManager, PerTurnTrainingValidator


def example_1_validate_saved_contexts():
    """Example 1: Validate that per-turn contexts are properly saved"""
    print("\n" + "="*80)
    print("Example 1: Validating Per-Turn Contexts")
    print("="*80)
    
    # Assume we have data from generation
    # In real usage, this comes from generation.py
    mock_data = create_mock_data_with_per_turn_contexts()
    
    # Validate
    results = PerTurnTrainingValidator.validate_context_match(mock_data, verbose=True)
    
    print(f"\nValidation Result: {'✅ Valid' if results['valid'] else '❌ Invalid'}")
    print(f"Number of trajectories: {results['num_trajectories']}")
    
    for traj_stat in results['trajectory_stats']:
        print(f"\nTrajectory {traj_stat['trajectory_id']}:")
        print(f"  Total turns: {traj_stat['num_turns']}")
        for turn in traj_stat['turns']:
            print(f"    Turn {turn['turn_id']}: "
                  f"context={turn['context_length']} tokens, "
                  f"response={turn['response_length']} tokens")


def example_2_extract_per_turn_contexts():
    """Example 2: Extract per-turn contexts from DataProto"""
    print("\n" + "="*80)
    print("Example 2: Extracting Per-Turn Contexts")
    print("="*80)
    
    mock_data = create_mock_data_with_per_turn_contexts()
    
    # Extract
    per_turn_contexts = PerTurnContextManager.extract_per_turn_contexts(mock_data)
    
    if per_turn_contexts:
        print(f"✅ Successfully extracted contexts for {len(per_turn_contexts)} trajectories")
        
        # Show first trajectory
        first_traj = per_turn_contexts[0]
        print(f"\nFirst trajectory has {len(first_traj)} turns:")
        for turn_idx, turn_ctx in enumerate(first_traj):
            print(f"  Turn {turn_idx}:")
            print(f"    Input IDs shape: {turn_ctx['input_ids'].shape}")
            print(f"    Context length: {turn_ctx['context_length']}")
            if 'response' in turn_ctx:
                print(f"    Response shape: {turn_ctx['response'].shape}")
    else:
        print("❌ No per-turn contexts found")


def example_3_compare_contexts():
    """Example 3: Compare original context with per-turn contexts"""
    print("\n" + "="*80)
    print("Example 3: Comparing Contexts (Show Context Mismatch)")
    print("="*80)
    
    mock_data = create_mock_data_with_per_turn_contexts()
    per_turn_contexts = PerTurnContextManager.extract_per_turn_contexts(mock_data)
    
    if per_turn_contexts:
        first_traj_contexts = per_turn_contexts[0]
        original_input_ids = mock_data.batch['input_ids'][0]
        step_ids = mock_data.batch['step_ids'][0]
        
        comparison = PerTurnTrainingValidator.compare_contexts(
            original_input_ids,
            first_traj_contexts,
            step_ids
        )
        
        print(f"Original (final compressed) context: {comparison['original_length']} tokens")
        print(f"Per-turn context lengths: {comparison['per_turn_lengths']}")
        print(f"\n{'⚠️  Context mismatch detected!' if comparison['context_mismatch_detected'] else '✅ Contexts match'}")
        
        if comparison['context_mismatch_detected']:
            print("\nThis means:")
            print("  - Without per-turn contexts: Early turns would use wrong context")
            print("  - With per-turn contexts: Each turn uses its actual context")
            print("  → Per-turn training fixes this mismatch!")


def example_4_integration_with_training():
    """Example 4: How to integrate per-turn training in actual training loop"""
    print("\n" + "="*80)
    print("Example 4: Integration with Training")
    print("="*80)
    
    print("""
# In your trainer code:

from verl.utils.per_turn_training import PerTurnContextManager

def compute_log_prob_wrapper(data: DataProto):
    \"\"\"
    Wrapper that uses per-turn contexts if available,
    otherwise falls back to original method.
    \"\"\"
    # Check if per-turn contexts are available
    per_turn_contexts = PerTurnContextManager.extract_per_turn_contexts(data)
    
    if per_turn_contexts is None:
        # Fallback to original method
        logger.info("Using original compressed context (no per-turn data)")
        return original_compute_log_prob(data)
    
    # Use per-turn contexts
    logger.info(f"Using per-turn contexts for {len(per_turn_contexts)} trajectories")
    return PerTurnContextManager.compute_per_turn_log_probs(
        data,
        compute_log_prob_fn=original_compute_log_prob,
        use_per_turn_context=True
    )

# Replace original compute_log_prob with wrapper
actor.compute_log_prob = compute_log_prob_wrapper
""")


def create_mock_data_with_per_turn_contexts():
    """Create mock data for examples"""
    import numpy as np
    
    # Mock trajectory with 3 turns
    # Turn 0: short context (initial question)
    # Turn 1: medium context (after first search)
    # Turn 2: compressed context (sliding window applied)
    
    per_turn_contexts = [[
        {
            'turn_id': 0,
            'input_ids': torch.randint(0, 1000, (1024,)),
            'attention_mask': torch.ones(1024),
            'context_length': 1024,
            'response': torch.randint(0, 1000, (256,)),
            'response_length': 256
        },
        {
            'turn_id': 1,
            'input_ids': torch.randint(0, 1000, (2048,)),
            'attention_mask': torch.ones(2048),
            'context_length': 2048,
            'response': torch.randint(0, 1000, (512,)),
            'response_length': 512
        },
        {
            'turn_id': 2,
            'input_ids': torch.randint(0, 1000, (1800,)),  # Compressed!
            'attention_mask': torch.ones(1800),
            'context_length': 1800,
            'response': torch.randint(0, 1000, (128,)),
            'response_length': 128
        }
    ]]
    
    # Mock DataProto
    data = DataProto.from_dict({
        'input_ids': torch.randint(0, 1000, (1, 2700)),  # Final compressed
        'responses': torch.randint(0, 1000, (1, 896)),  # All responses concatenated
        'step_ids': torch.tensor([[0]*256 + [1]*512 + [2]*128])  # Which turn each token belongs to
    })
    
    # Add per_turn_contexts to non_tensor_batch
    data.non_tensor_batch['per_turn_contexts'] = np.array(per_turn_contexts, dtype=object)
    
    return data


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Per-Turn Training Examples")
    print("="*80)
    print("\nThese examples demonstrate how to use per-turn context saving")
    print("to fix the context mismatch problem in multi-turn training.")
    
    # Run examples
    example_1_validate_saved_contexts()
    example_2_extract_per_turn_contexts()
    example_3_compare_contexts()
    example_4_integration_with_training()
    
    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run training with per-turn contexts enabled")
    print("  2. Monitor PPO ratios for each turn (should be normal for all turns)")
    print("  3. Compare with baseline (should see improvement in early turns)")
    print("\nFor more details, see docs/PER_TURN_TRAINING_GUIDE.md")


