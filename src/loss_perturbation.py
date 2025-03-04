def get_perturbed_loss(
    current_loss: float,
    current_epoch: int,
    n_loss_perturbation_epochs: int,
    expected_loss_after_perturbation_epochs: float,
) -> float:
    current_epoch_t: float = current_epoch / n_loss_perturbation_epochs
    current_linear_loss: float = (1 - current_epoch_t) * 1 + current_epoch_t * expected_loss_after_perturbation_epochs
    current_perturbed_loss: float = (1 - current_epoch_t) * current_linear_loss + current_epoch_t * current_loss
    return current_perturbed_loss
