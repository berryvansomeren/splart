import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Run with "streamlit run ./dev_tools/ui_simulate_loss.py"

# Streamlit App Title
st.title("Perturbed Loss Curve Simulator")

"""
In the Splart config you will find a :orange[n_loss_perturbation_epochs] variable. 
This variable is used to determine how many epochs will be perturbed in the loss curve. 
In this simulation we will show how the loss curve changes when perturbed.  

The reason we want to perturb the loss is that at the start the loss is 1.0,
and after just 5 epochs it is already down to 0.3 - 0.4.
This results in a big size difference between brushes at the start of the optimization.
To have a more gradual change in brush size, we perturb the loss as used to determine the brush size. 
A nice additional benefit is that it also smoothes out the loss curve a bit. 
:orange[Note that we do not actually change the loss for backpropagation, 
only the way it is used for rendering purposes.]

Perturbation is achieved by first defining for how many epochs we want to perturb the loss, 
and then defining what we expect the natural loss to be at that point. 
This expected value does not need to be very accurate. 
A :red[linear loss curve] is then created by starting at *(0,1)* and drawing a line to *(t,L_expected)*.
For every epoch we then interpolate between the :blue[simulated loss curve] and the :red[linear loss curve],
to obtain the final :green[perturbed loss curve].
"""

# Sidebar Controls for Parameters
st.sidebar.header("Simulation Parameters")
epochs = st.sidebar.slider("Number of Epochs", 10, 500, 200, 10)
initial_loss = st.sidebar.slider("Initial Loss", 0.1, 1.0, 0.4, 0.1)  # Default set to 0.3
decay_rate = st.sidebar.slider("Decay Rate", 0.0, 0.10, 0.01, 0.01)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.1, 0.01, 0.01)

# Generate Epochs
epoch_range = np.arange(0, epochs, 1)

# Simulate Loss Function (Exponential Decay)
simulated_loss = initial_loss * np.exp(-decay_rate * epoch_range) + np.random.normal(
    0, noise_level, size=len(epoch_range)
)

# Slider to choose t for the straight lines
t = st.sidebar.slider("Number of epochs for loss perturbation", 0, epochs, epochs // 2, 1)
# Slider to choose t for the straight lines
expected_loss_at_t = st.sidebar.slider("Expected loss when loss perturbation ends", 0.0, 1.0, 0.4, 0.1)

# Linear Loss: From (0,1) to (t, L(t))
linear_loss_limits_xs = [0, t]
linear_loss_limits_ys = [1, expected_loss_at_t]

# Interpolated Loss: Interpolating between the linear loss and L(t), stopping at t
sample_xs = np.linspace(0, t, t)
linear_loss_ys = (1 - sample_xs / t) * 1 + (sample_xs / t) * expected_loss_at_t
perturbed_loss_ys = (1 - sample_xs / t) * linear_loss_ys + (sample_xs / t) * simulated_loss[:t]

# Plot the Loss Curve
fig, ax = plt.subplots()
ax.plot(epoch_range, simulated_loss, label="Simulated Loss Curve", color="blue")
ax.plot(sample_xs, linear_loss_ys, label="Linear Loss Curve", color="red", linestyle="dotted")
ax.plot(sample_xs, perturbed_loss_ys, label="Perturbed Loss Curve", color="green", linestyle="--")

# Labels and Legend
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve with Dynamic Interpolation")
ax.legend()
ax.grid(True)

# Display the Plot in Streamlit
st.pyplot(fig)
