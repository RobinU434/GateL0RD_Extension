import numpy as np

EXPONENTIAL_DEFAULTS = {
    "slope": 0.1,
    "start_steps": 5,
}

LINEAR_DEFAULTS = {
    "total_epochs": 100,
    "start_steps": 5,
}

INVERSE_SIGMOID_DEFAULTS = {
    "total_epochs": 100,
    "start_steps": 5,
}


def exponential_scheduled_sampling(
    epoch: int, slope: float, seq_len: int, batch_size: int, start_steps: int = 5
):
    """
    Exponential decay from teacher forcing to free-running

    Args:
        epoch (int): current training epoch
        slope (float): decay rate
        seq_len (int): sequence length
        batch_size (int): batch size
        start_steps (int): number of initial teacher forcing steps

    Returns:
        schedule: [seq_len, batch_size] probability matrix
    """
    schedule = np.ones((seq_len, batch_size))

    # Always use teacher forcing for first few steps
    if start_steps > 0:
        schedule[:start_steps] = 1.0

    # Exponential decay for remaining steps
    for t in range(start_steps, seq_len):
        # Probability of using ground truth decreases exponentially
        prob = np.exp(-slope * epoch * (t - start_steps + 1) / (seq_len - start_steps))
        schedule[t] = np.clip(prob, 0.0, 1.0)

    return schedule


def linear_scheduled_sampling(
    epoch: int, total_epochs: int, seq_len: int, batch_size: int, start_steps: int = 5
):
    """
    Linear decay from teacher forcing to free-running

    Args:
        epoch (int): current training epoch
        total_epochs (int): total number of training epochs
        seq_len (int): sequence length
        batch_size (int): batch size
        start_steps (int): number of initial teacher forcing steps

    Returns:
        schedule: [seq_len, batch_size] probability matrix
    """
    schedule = np.ones((seq_len, batch_size))

    # Always use teacher forcing for first few steps
    if start_steps > 0:
        schedule[:start_steps] = 1.0

    # Linear decay for remaining steps
    for t in range(start_steps, seq_len):
        prob = 1.0 - (epoch / total_epochs) * (
            (t - start_steps + 1) / (seq_len - start_steps)
        )
        schedule[t] = np.clip(prob, 0.0, 1.0)

    return schedule


def inverse_sigmoid_scheduled_sampling(
    epoch: int, total_epochs: int, seq_len: int, batch_size: int, start_steps: int = 5
):
    """
    Inverse sigmoid decay from teacher forcing to free-running

    Args:
        epoch (int): current training epoch
        total_epochs (int): total number of training epochs
        seq_len (int): sequence length
        batch_size (int): batch size
        start_steps (int): number of initial teacher forcing steps

    Returns:
        schedule: [seq_len, batch_size] probability matrix
    """
    schedule = np.ones((seq_len, batch_size))

    # Always use teacher forcing for first few steps
    if start_steps > 0:
        schedule[:start_steps] = 1.0

    # Inverse sigmoid decay for remaining steps
    for t in range(start_steps, seq_len):
        prob = 1.0 / (
            1.0
            + np.exp(
                -10
                * (
                    epoch / total_epochs
                    - (t - start_steps + 1) / (seq_len - start_steps)
                )
            )
        )
        schedule[t] = np.clip(prob, 0.0, 1.0)

    return schedule


def sample_schedule(schedule: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
    """Sample from the schedule to get a binary mask for teacher forcing.

    Args:
        schedule (np.ndarray): [seq_len, batch_size] probability matrix
        batch_size (int): batch size
        seq_len (int): sequence length

    Returns:
        np.ndarray: [seq_len, batch_size] binary mask
    """
    # Sample from the schedule to get a binary mask
    mask = np.random.rand(seq_len, batch_size) < schedule
    return mask.astype(np.float32)
