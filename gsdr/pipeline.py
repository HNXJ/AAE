import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from .analysis import calculate_firing_rates, compute_psd, calculate_mcdp, calculate_psd_bands, compute_unscaled_psd_from_trace
from .simulation import noise_current_ac

def get_loss_fn(net, transform, dt_global, global_psd_interval, 
                lower_c, upper_c, firing_rate_weight, psd_weight,
                num_e, checkpoints):
    """
    Returns a configured loss function.
    """
    def simulate_wrapper(params, input_amp):
        ac_currents = noise_current_ac(
            i_delay=500.0, i_dur=500.0, amp_n=0.0, amp_b=input_amp,
            spect=jnp.array([120.0]), delta_t=dt_global, t_max=1500.0
        )
        net.delete_stimuli()
        data_stimuli = net.cell(list(range(0, num_e, 2))).branch(1).loc(0.0).data_stimulate(ac_currents)
        return jx.integrate(net, params=params, data_stimuli=data_stimuli, checkpoint_lengths=checkpoints)

    batched_simulate = jax.vmap(simulate_wrapper, in_axes=(None, 0))

    def loss_fn(opt_params, inputs, labels):
        params = transform.forward(opt_params)
        traces = batched_simulate(params, inputs)

        def compute_psd_scaled(trace):
            # Focus on [100ms, 1400ms]
            t_l, t_r = int(100/dt_global), int(1400/dt_global)
            signal = jnp.mean(trace[:, t_l:t_r], axis=0)
            _, psd = compute_psd(signal, dt_global, target_freqs=global_psd_interval)
            return psd / (jnp.max(psd) + 1e-6)

        predictions = jax.vmap(compute_psd_scaled)(traces)
        psd_loss = jnp.mean(jnp.sum(jnp.square(labels * jnp.square(predictions - labels)), axis=1))

        firing_rates = calculate_firing_rates(traces, dt_global)
        penalty = jnp.mean(jnp.exp(lower_c - firing_rates) + jnp.exp(firing_rates - upper_c))
        
        total_loss = psd_loss * psd_weight + penalty * firing_rate_weight
        return total_loss, traces

    return loss_fn

def train_net(net, optimizer, transform, dataloader, loss_fn, 
              ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds,
              dt_global, band_definitions, epoch_n=100):
    """
    Main training loop for the NetEIG model using GSDR.
    """
    opt_params = net.get_parameters()
    opt_state = optimizer.init(opt_params)
    key = jax.random.PRNGKey(0)
    jitted_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    training_log = {k: [] for k in ["loss", "alpha", "avg_gAMPA", "avg_gGABAa", "gamma", "beta", "alpha_band", "theta"]}

    for epoch in range(epoch_n):
        key, step_key = jax.random.split(key)
        for batch in dataloader:
            inputs, labels = batch
            (loss_val, traces), grads = jitted_grad(opt_params, inputs, labels)
            
            if jnp.isnan(loss_val):
                mcdp_factors = None
            else:
                mcdp_factors = calculate_mcdp(traces, ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds)

            updates, opt_state = optimizer.update(grads, opt_state, params=transform.forward(opt_params), 
                                                 value=loss_val, key=step_key, mcdp_factors=mcdp_factors)
            opt_params = optax.apply_updates(opt_params, updates)

        # Logging logic (simplified for package)
        params_f = transform.forward(opt_params)
        print(f"Epoch {epoch}: Loss {loss_val:.4f}")
        training_log["loss"].append(float(loss_val))

    return transform.forward(opt_params), training_log
