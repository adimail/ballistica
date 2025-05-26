# Released under the MIT License. See LICENSE for details.

from __future__ import annotations

from typing import TYPE_CHECKING

import bascenev1 as bs

if TYPE_CHECKING:
    pass


def calculate_reward(
    game_activity: bs.GameActivity,
    agent_spaz: bs.Actor | None,
    prev_agent_hp: int,
    current_agent_hp: int,
    prev_agent_kills: int,
    current_agent_kills: int,
    prev_opponent_hp: dict[int, int],  # bot_id -> hp
    current_opponent_hp: dict[int, int],  # bot_id -> hp
    game_over: bool,
    agent_won: bool,
) -> float:
    """
    Calculates the reward for the agent based on game state changes.
    """
    reward = 0.0

    # Reward for staying alive (small positive reward per step)
    if agent_spaz and agent_spaz.is_alive():
        reward += 0.01

    # Penalty for taking damage
    damage_taken = prev_agent_hp - current_agent_hp
    if damage_taken > 0:
        reward -= damage_taken * 0.05  # Penalize per HP point lost

    # Reward for dealing damage
    for bot_id, hp in current_opponent_hp.items():
        if bot_id in prev_opponent_hp:
            damage_dealt = prev_opponent_hp[bot_id] - hp
            if damage_dealt > 0:
                reward += damage_dealt * 0.1  # Reward per HP point dealt

    # Reward for kills
    kills_made = current_agent_kills - prev_agent_kills
    if kills_made > 0:
        reward += kills_made * 20.0

    # Penalty for dying (if agent_spaz was alive and now is not)
    if prev_agent_hp > 0 and current_agent_hp <= 0:
        reward -= 50.0  # Large penalty for dying

    # Large reward/penalty for game outcome
    if game_over:
        if agent_won:
            reward += 200.0
        else:
            # Check if it was a draw or loss. For now, any non-win is a penalty.
            # You might want to differentiate draws.
            reward -= 100.0
            # If the agent died and lost, it's already penalized for dying.
            # If it was alive but the team lost (e.g. time up), this applies.

    return reward
