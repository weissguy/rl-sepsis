import flax
import jax
import jax.numpy as jnp
import numpy as np
from utils import PRNGKey, TrainState, Batch, InfoDict, target_update



def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey # type: ignore
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @jax.jit
    def update(agent, batch: Batch) -> InfoDict:
        def critic_loss_fn(critic_params):
            next_v = agent.value(batch['next_observations'])
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v
            q1, q2 = agent.critic(batch['observations'], batch['actions'], params=critic_params)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            return critic_loss, {
                'critic_loss': critic_loss,
                'q1': q1.mean(),
            }
        
        def value_loss_fn(value_params):
            q1, q2 = agent.target_critic(batch['observations'], batch['actions'])
            q = jnp.minimum(q1, q2)
            v = agent.value(batch['observations'], params=value_params)
            value_loss = expectile_loss(q-v, agent.config['expectile']).mean()
            return value_loss, {
                'value_loss': value_loss,
                'v': v.mean(),
            }
        
        def actor_loss_fn(actor_params):
            v = agent.value(batch['observations'])
            q1, q2 = agent.critic(batch['observations'], batch['actions'])
            q = jnp.minimum(q1, q2)
            exp_a = jnp.exp((q - v) * agent.config['temperature'])
            clip_ratio = jnp.mean(exp_a > 100.0)
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = agent.actor(batch['observations'], params=actor_params)
            log_probs = dist.log_prob(batch['actions'])
            actor_loss = -(exp_a * log_probs).mean()

            return actor_loss, {'actor_loss': actor_loss, 'adv': q - v, 'clip_ratio': clip_ratio}
        
        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        return agent.replace(critic=new_critic, target_critic=new_target_critic, value=new_value, actor=new_actor), {
            **critic_info, **value_info, **actor_info
        }

    @jax.jit
    def sample_actions(agent,
                       observations: np.ndarray,
                       *,
                       seed: PRNGKey, # type: ignore
                       temperature: float = 1.0) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions