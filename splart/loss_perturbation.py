from splart.training_config import TrainingConfig


def _lerp(start: float, end: float, t: float) -> float:
    """
    Linear interpolation between start and end values based on parameter t (0 to 1).
    """
    return start * (1 - t) + end * t


def _get_interpolated_loss(
    config: TrainingConfig,
    current_loss: float,
    current_epoch: int,
) -> float:
    """
    See dev_tools/ui_simulate_loss.py to get a better felling for how the loss is affected
    By creating a linear target loss (created by lerping, but it's just a line),
    and then lerping between that line and the actual loss,
    we create a perturbed loss curve
    """
    epoch_t: float = min(1.0, current_epoch / config.n_loss_interpolation_epochs)
    linear_target_loss: float = _lerp(1.0, config.expected_loss_after_perturbation_epochs, epoch_t)
    interpolated_loss: float = _lerp(linear_target_loss, current_loss, epoch_t)
    return interpolated_loss


def get_perturbed_loss(
    config: TrainingConfig,
    current_loss: float,
    current_epoch: int,
) -> float:
    perturbed_loss = min(1.0, current_loss * config.base_loss_multiplier )
    if current_epoch < config.n_loss_interpolation_epochs:
        interpolated_perturbed_loss = _get_interpolated_loss(
            config=config,
            current_loss=perturbed_loss,
            current_epoch=current_epoch,
        )
        perturbed_loss = interpolated_perturbed_loss
    return perturbed_loss
