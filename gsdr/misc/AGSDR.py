import jax
import jax.numpy as jnp
import jaxley as jx
import optax
from typing import Callable, Any
from flax.struct import dataclass

@dataclass
class GSDRState:
    inner_state: Any
    params_opt: Any
    inner_state_opt: Any
    loss_opt: float
    a: float
    a_opt: float
    lambda_d: float
    step_count: int
    consecutive_unchanged_epochs: int
    last_optimal_change_step: int
    var_sup_ema: float = 1.0
    var_unsup_ema: float = 1.0

def AGSDR(
    inner_optimizer: optax.GradientTransformation,
    delta_distribution: Callable = jax.random.normal,
    deselection_threshold: float = 2.0,
    a_init: float = 0.5,
    lambda_d: float = 1.0,
    checkpoint_n: int = 10,
    tau_a_growth: float = 10.0,
    mcdp: bool = True,
    ema_momentum: float = 0.9,
    alpha_min: float = 0.1,
    alpha_max: float = 0.9
) -> optax.GradientTransformation:
    """
    Adaptive GSDR (AGSDR) v2.5 with 'Alpha Jolt' protocol.
    Forces alpha=0 for 1 trial after state change to stabilize gradients.
    """
    def init_fn(params):
        inner_state = inner_optimizer.init(params)
        return GSDRState(
            inner_state=inner_state,
            params_opt=params,
            inner_state_opt=inner_state,
            loss_opt=jnp.inf,
            a=a_init,
            a_opt=a_init,
            lambda_d=lambda_d,
            step_count=0,
            consecutive_unchanged_epochs=0,
            last_optimal_change_step=0,
            var_sup_ema=1.0,
            var_unsup_ema=1.0
        )

    def update_fn(updates, state, params=None, value=None, key=None, mcdp_factors=None):
        if params is None or value is None or key is None:
            raise ValueError("AGSDR requires 'params', 'value' (loss), and 'key'.")

        grads = updates
        loss = value

        is_new_opt = (loss < state.loss_opt)
        new_params_opt = jax.tree.map(lambda cur, opt: jnp.where(is_new_opt, cur, opt), params, state.params_opt)
        new_loss_opt = jnp.where(is_new_opt, loss, state.loss_opt)
        new_inner_state_opt = jax.tree.map(lambda cur, opt: jnp.where(is_new_opt, cur, opt), state.inner_state, state.inner_state_opt)

        next_consecutive_unchanged_epochs = jnp.where(is_new_opt, 0, state.consecutive_unchanged_epochs + 1)
        step_of_last_optimal_change = jnp.where(is_new_opt, state.step_count + 1, state.last_optimal_change_step)

        is_deselect = ((loss > (new_loss_opt * deselection_threshold)) & (new_loss_opt != jnp.inf)) | (jnp.isnan(loss))
        is_reset_due_to_checkpoint = (state.step_count > 0) & (next_consecutive_unchanged_epochs >= checkpoint_n) & (new_loss_opt != jnp.inf)
        should_reset = is_deselect | is_reset_due_to_checkpoint

        def reset_branch(operand):
            _params, _new_params_opt, _new_inner_state_opt, _current_step = operand
            jump_delta = jax.tree.map(lambda opt_p, cur_p: opt_p - cur_p, _new_params_opt, _params)
            reset_state = state.replace(
                inner_state=_new_inner_state_opt, params_opt=_new_params_opt, 
                inner_state_opt=_new_inner_state_opt, loss_opt=new_loss_opt,
                step_count=_current_step, consecutive_unchanged_epochs=0,
                last_optimal_change_step=_current_step
            )
            return jump_delta, reset_state

        def normal_branch(operand):
            _params, _new_params_opt, _new_inner_state_opt, _current_step = operand
            time_since_last_change = jnp.maximum(0, _current_step - step_of_last_optimal_change)
            
            # --- ALPHA JOLT Protocol ---
            is_jolt_epoch = (time_since_last_change == 0)
            
            effective_lambda_d = (time_since_last_change**2) * (1.0 - jnp.exp(-(time_since_last_change) / tau_a_growth))

            inner_opt_key, noise_key = jax.random.split(key, 2)
            inner_updates, updated_inner_state = inner_optimizer.update(grads, state.inner_state, _params, key=inner_opt_key)

            param_leaves, treedef = jax.tree.flatten(_params)
            subkeys = jax.random.split(noise_key, len(param_leaves))
            delta_d = jax.tree.map(lambda p, k: delta_distribution(k, p.shape), _params, jax.tree.unflatten(treedef, subkeys))

            if mcdp and mcdp_factors is not None:
                delta = jax.tree.map(lambda n, p, r: n * p * r, delta_d, _params, mcdp_factors)
            else:
                delta = jax.tree.map(lambda n, p: n * p, delta_d, _params)

            flat_inner = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(inner_updates)])
            flat_delta = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(delta)])
            curr_var_sup = jnp.var(flat_inner)
            curr_var_unsup = jnp.var(flat_delta)
            
            new_var_sup_ema = ema_momentum * state.var_sup_ema + (1 - ema_momentum) * curr_var_sup
            new_var_unsup_ema = ema_momentum * state.var_unsup_ema + (1 - ema_momentum) * curr_var_unsup
            
            epsilon = 1e-8
            denom = new_var_sup_ema + new_var_unsup_ema + epsilon
            next_a = jnp.where(denom > 1e-6, new_var_sup_ema / denom, 0.8)
            
            # Apply Jolt: next_a=0 if is_jolt_epoch, otherwise clipped to floor
            next_a = jnp.where(is_jolt_epoch, 0.0, jnp.clip(next_a, alpha_min, alpha_max))
            
            combined_updates = jax.tree.map(lambda d, g: effective_lambda_d * (next_a * d + (1.0 - next_a) * g), delta, inner_updates)

            return combined_updates, GSDRState(
                inner_state=updated_inner_state, params_opt=_new_params_opt,
                inner_state_opt=_new_inner_state_opt, loss_opt=new_loss_opt,
                a=next_a, a_opt=jnp.where(is_new_opt, next_a, state.a_opt), 
                lambda_d=state.lambda_d,
                step_count=_current_step, consecutive_unchanged_epochs=next_consecutive_unchanged_epochs,
                last_optimal_change_step=step_of_last_optimal_change,
                var_sup_ema=new_var_sup_ema, var_unsup_ema=new_var_unsup_ema
            )

        current_step = state.step_count + 1
        return jax.lax.cond(should_reset, reset_branch, normal_branch, (params, new_params_opt, new_inner_state_opt, current_step))

    return optax.GradientTransformation(init_fn, update_fn)
