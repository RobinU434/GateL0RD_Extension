import logging
from typing import List

import torch


def get_devices(gpus: int, logger: logging.Logger) -> List[int] | str:
    """find the number of gpus available

    Args:
        gpus (int): number of gpus needed for the run
        logger (Logger): experiment logger
    Returns:
        List[int]: list of gpu devices or cpu
    """

    # check if cuda is available
    if not torch.cuda.is_available():
        if gpus > 0:
            logger.warning("No GPUs available. Falling back to CPU")
        return "auto"

    # find the the number of gpus
    num_gpus = torch.cuda.device_count()

    # choose the minimum between number of gpus wanted and gpus avaialble
    if gpus > num_gpus:
        logger.warning("Requested more GPUs than available. Using %d GPUs", num_gpus)
    return [i for i in range(min(gpus, num_gpus))]
