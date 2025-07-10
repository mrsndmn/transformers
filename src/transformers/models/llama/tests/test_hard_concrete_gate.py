import torch
import pytest
from transformers.models.llama.modeling_adaptive_llama import HardConcreteGate

@pytest.fixture
def default_gate():
    return HardConcreteGate(count_log_a=10, max_seq_len=128, log_a=0.0,)

@pytest.fixture
def custom_gate():
    return HardConcreteGate(
        count_log_a=10,
        log_a=1.0,
        max_seq_len=64,
        temperature=0.5,
        learnt_temperature=False,
        adjust_range=(-0.2, 1.2),
        eps=1e-6
    )

def test_initialization_default(default_gate):
    assert default_gate.hcg_log_a.shape == (10,)
    assert torch.all(default_gate.hcg_log_a == 0.0)
    assert torch.allclose(default_gate.temperature, torch.tensor(0.33))
    assert torch.allclose(default_gate.adjust_range, torch.tensor([-0.1, 1.1]))
    assert default_gate.eps == 1e-9
    assert default_gate.random_buffer.shape == (1, 128, 1)

def test_initialization_custom(custom_gate):
    assert custom_gate.hcg_log_a.shape == (10,)
    assert torch.all(custom_gate.hcg_log_a == 1.0)
    assert custom_gate.temperature.item() == 0.5
    assert torch.allclose(custom_gate.adjust_range, torch.tensor([-0.2, 1.2]))
    assert custom_gate.eps == 1e-6
    assert custom_gate.random_buffer.shape == (1, 64, 1)

@pytest.mark.parametrize("batch_size, seq_len", [(2, 50), (1, 128)])
def test_get_p_open(default_gate, batch_size, seq_len):
    input_ids = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    p_open = default_gate.get_p_open(input_ids)
    assert p_open.shape == (batch_size, seq_len, 1)
    assert torch.all((p_open >= default_gate.eps) & (p_open <= 1.0 - default_gate.eps))

@pytest.mark.parametrize("batch_size, seq_len", [(2, 50), (1, 128)])
def test_forward_training(default_gate, batch_size, seq_len):
    default_gate.train()
    input_ids = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    # Add some padding
    if seq_len > 10:
        attention_mask[:, -10:] = 0

    concrete = default_gate(input_ids, attention_mask)
    assert concrete.shape == (batch_size, seq_len, 1)
    assert torch.all((concrete >= 0.0) & (concrete <= 1.0))
    # Check masked values
    assert torch.all(concrete[attention_mask == 0] == 0)
    # Check unmasked values are potentially non-zero (probabilistic)
    if torch.any(attention_mask == 1):
        assert torch.any(concrete[attention_mask == 1] > 0)

@pytest.mark.parametrize("batch_size, seq_len", [(2, 50), (1, 128)])
def test_forward_eval(default_gate, batch_size, seq_len):
    default_gate.eval()
    input_ids = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    # Add some padding
    if seq_len > 10:
        attention_mask[:, -10:] = 0

    concrete1 = default_gate(input_ids, attention_mask)
    concrete2 = default_gate(input_ids, attention_mask) # Should be deterministic in eval

    assert concrete1.shape == (batch_size, seq_len, 1)
    assert torch.all((concrete1 >= 0.0) & (concrete1 <= 1.0))
    assert torch.allclose(concrete1, concrete2)
    # Check masked values
    assert torch.all(concrete1[attention_mask == 0] == 0)

    # Check calculation against expected formula in eval
    log_a = default_gate.hcg_log_a[input_ids].unsqueeze(-1)
    expected_concrete = torch.sigmoid(log_a)
    expected_concrete = expected_concrete * (default_gate.adjust_range[1] - default_gate.adjust_range[0]) + default_gate.adjust_range[0]
    expected_concrete = torch.clip(expected_concrete, min=0, max=1)
    expected_concrete[attention_mask == 0] = 0
    assert torch.allclose(concrete1, expected_concrete)

def test_attention_mask_effect(default_gate):
    batch_size, seq_len = 1, 10
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long) # Use zero IDs for simplicity
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.long)

    # Training mode
    default_gate.train()
    concrete_train = default_gate(input_ids, attention_mask)
    assert torch.all(concrete_train[attention_mask == 0] == 0)
    assert torch.all(concrete_train[0, [3, 4, 6, 9]] == 0)
    # Check that at least some unmasked values are non-zero (highly probable)
    assert torch.any(concrete_train[0, [0, 1, 2, 5, 7, 8]] > 0)

    # Eval mode
    default_gate.eval()
    concrete_eval = default_gate(input_ids, attention_mask)
    assert torch.all(concrete_eval[attention_mask == 0] == 0)
    assert torch.all(concrete_eval[0, [3, 4, 6, 9]] == 0)

    # Calculate expected eval value for unmasked tokens
    log_a = default_gate.hcg_log_a[input_ids[0, 0]] # All unmasked have ID 0
    expected_val = torch.sigmoid(log_a)
    expected_val = expected_val * (default_gate.adjust_range[1] - default_gate.adjust_range[0]) + default_gate.adjust_range[0]
    expected_val = torch.clip(expected_val, min=0, max=1)

    assert torch.allclose(concrete_eval[0, [0, 1, 2, 5, 7, 8]], expected_val)

# Add more tests if specific edge cases or behaviors need verification
# For example, testing with different dtypes if necessary, or testing gradients.
# Testing learnt_temperature=True would require a separate setup and possibly training steps.
