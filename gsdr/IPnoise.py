import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from typing import Optional

class IPnoise(jx.channels.Channel):
    """
    Impulse Poisson Noise Channel.
    Generates discrete current pulses with Poisson-distributed intervals.
    """
    def __init__(self, name: str = None, seed: Optional[int] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "pulse_width": 0.1, 
            "pulse_amp": 1.0, 
            "poisson_l": 50.0,
            "seed": float(seed) if seed is not None else float(np.random.randint(0, 2**31 - 1))
        }
        self.channel_states = {"last_pulse_t": -1000.0, "next_interval": 50.0, "step": 0.0}
        self.current_name = "i_impulse_poisson"

    def update_states(self, states, dt, v, params):
        t = states["step"] * dt
        last_t = states["last_pulse_t"]
        curr_interval = states["next_interval"]
        
        # Trigger next pulse if time elapsed
        should_trigger = (t >= (last_t + curr_interval))
        
        seeds_int = params["seed"].astype(int)
        
        if seeds_int.ndim == 0:
            base_key = jax.random.PRNGKey(seeds_int)
            step_key = jax.random.fold_in(base_key, states["step"].astype(int))
            raw_rand = jax.random.uniform(step_key, minval=1e-7, maxval=1.0)
        else:
            base_key = jax.vmap(jax.random.PRNGKey)(seeds_int)
            step_key = jax.vmap(jax.random.fold_in)(base_key, states["step"].astype(int))
            raw_rand = jax.vmap(lambda k: jax.random.uniform(k, minval=1e-7, maxval=1.0))(step_key)
        
        new_interval = -params["poisson_l"] * jnp.log(raw_rand)
        new_interval = jnp.clip(new_interval, 10.0, 10.0 * params["poisson_l"])
        
        next_last_t = jnp.where(should_trigger, t, last_t)
        next_interval = jnp.where(should_trigger, new_interval, curr_interval)
        
        return {
            "last_pulse_t": next_last_t,
            "next_interval": next_interval,
            "step": states["step"] + 1.0
        }

    def compute_current(self, states, v, params):
        # We need the dt here too, but jaxley doesn't pass it to compute_current easily
        # Use a safe approximation or assume 0.1
        dt_approx = 0.1 
        t = states["step"] * dt_approx
        
        # Pulse is active if current time is within [last_t, last_t + width]
        # We add a small epsilon to ensure the window is inclusive of the sample
        is_active = (t >= states["last_pulse_t"]) & \
                    (t < (states["last_pulse_t"] + params["pulse_width"] + 1e-5))
        
        current = jnp.where(is_active, params["pulse_amp"], 0.0)
        return -current

    def init_state(self, states, v, params, delta_t):
        # Stagger initialization to prevent a 'First-Sample Spark'
        # Generate a random initial last_pulse_t between -poisson_l and 0
        key = jax.random.PRNGKey(params["seed"].astype(int))
        stagger = jax.random.uniform(key, minval=-params["poisson_l"], maxval=0.0)
        return {"last_pulse_t": stagger, "next_interval": params["poisson_l"], "step": 0.0}
