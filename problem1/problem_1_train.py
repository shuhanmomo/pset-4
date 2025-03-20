"""
Implement a training loop for the MLP and SIREN models.
"""

import torch
from torch.utils.data import DataLoader

from problem_1_gradients import gradient, laplace
from problem_1_mlp import MLP
from problem_1_siren import SIREN
from problem1_grid import GRID
from utils import ImageDataset, plot, psnr, set_seed
from typing import Dict, Any


def train(
    model,  # "MLP" or "SIREN"
    dataset: ImageDataset,  # Dataset of coordinates and pixels for an image
    lr: float,  # Learning rate
    total_steps: int,  # Number of gradient descent step
    steps_til_summary: int,  # Number of steps between summaries (i.e. print/plot)
    device: torch.device,  # "cuda" or "cpu"
    seed: int = 245,
    plot_multi_models: bool = False,
    **kwargs: Dict[str, Any],  # Model-specific arguments
):
    """
    Train the model on the provided dataset.

    Given the **kwargs, initialize a neural field model and an optimizer.
    Then, train the model and log the loss and PSNR for each step. Examples
    in the notebook use MSE loss, but feel free to experiment with other
    objective functions. Additionally, in the notebook, we plot the reconstruction
    and various gradients every `steps_til_summary` steps using `utils.plot()`.

    You re allowed to change the arguments as you see fit so long as you can plot
    images of the reconstruction and the gradients/laplacian every `steps_til_summary` steps.
    Look at `should_look_like` for examples of what we would like to see. Make sure to
    also plot (MSE) loss and PSNR every `steps_til_summary` steps.

    You should train for `total_steps` gradient steps on the whole image (look at `ImageDataset` in `utils.py`)
    and visualize the results every `steps_til_summary` steps. The visualization must at least include:
    1. The MSE and PSNR
    2. The reconstructed image
    (Optionally you can also include the laplace or gradient of the image).

    PSNR is defined here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    set_seed(seed)
    # initialize model
    model_type = model
    if model_type == "MLP":
        model = MLP(**kwargs)
    elif model_type == "SIREN":
        model = SIREN(**kwargs)
    elif model_type == "GRID":
        model = GRID(**kwargs)
    model.to(device)
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss
    loss_fn = torch.nn.MSELoss()
    mse_history = []
    psnr_history = []

    # start iterations
    for iteration in range(total_steps):
        optimizer.zero_grad()
        samples = dataset.coords.to(device)
        predicted, _ = model(samples)
        ground_truth = dataset.pixels.to(device)

        if (iteration + 1) % steps_til_summary == 0:

            if model_type == "GRID":  # gradient for grid sample is not implemented
                grad = torch.zeros_like(predicted)
                laplacian = torch.zeros_like(predicted)
            else:
                grad = gradient(predicted, samples)
                laplacian = laplace(predicted, samples)

        loss = loss_fn(predicted, ground_truth)
        loss.backward(retain_graph=True)
        optimizer.step()

        # record
        mse_val = loss.item()
        mse_history.append(mse_val)
        psnr_score = psnr(predicted, ground_truth).cpu().detach().float()
        psnr_history.append(psnr_score)

        # plot images
        if (iteration + 1) % steps_til_summary == 0 and not plot_multi_models:
            # model_outputs = predicted.clone().detach().requires_grad_(True)
            print(f"Step {iteration} : loss={mse_val:4f}, psnr={psnr_score:4f}")
            plot(dataset, predicted, grad, laplacian)

    if not plot_multi_models:
        return mse_history, psnr_history
    else:
        return mse_history, psnr_history, predicted, grad, laplacian
