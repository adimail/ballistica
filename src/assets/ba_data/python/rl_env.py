# Released under the MIT License. See LICENSE for details.


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, cast

import bascenev1 as bs
import bauiv1 as bui
import gymnasium
import numpy as np
import rl_reward
from bascenev1lib.actor.playerspaz import PlayerSpaz
from bascenev1lib.actor.spazbot import SpazBot
from bascenev1lib.game.deathmatch import DeathMatchGame
from gymnasium import spaces


class DeathMatchEnv(gymnasium.Env[np.ndarray, np.ndarray]):
    """
    Custom Environment for BombSquad DeathMatch that follows the Gymnasium API.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Define actions (adjust as needed)
    ACTION_NOOP = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2
    ACTION_MOVE_UP = 3
    ACTION_MOVE_DOWN = 4
    ACTION_JUMP = 5
    ACTION_PUNCH = 6
    ACTION_BOMB = 7
    ACTION_PICKUP = 8
    # For combined movement/action, you'd need MultiDiscrete or Box space
    # Example: Move FWD + Punch, Move BWD + Bomb etc.
    # For now, one action at a time. Movement actions will "stick" until
    # another movement action or NOOP for movement is chosen.

    def __init__(
        self,
        map_name: str,
        render_mode: str | None = "human",
        training_activity: Any = None,
    ):
        super().__init__()
        self.map_name = map_name
        self.render_mode = render_mode
        self._training_activity = (
            training_activity  # Keep a ref to the activity running the training
        )

        # Define action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)

        # Define observation space (example, adjust to your needs)
        # Agent: pos(3), vel(3), hp(1), bomb_count(1) = 8
        # N closest enemies: pos(3*N), vel(3*N), hp(N) = 7*N
        # N closest bombs: pos(3*N), timer(N) = 4*N
        # N closest powerups: pos(3*N) = 3*N
        # For N=3: 8 + 21 + 12 + 9 = 50 features
        self.num_closest_enemies = 3
        self.num_closest_bombs = 3
        self.num_closest_powerups = 3

        obs_size = (
            8
            + (7 * self.num_closest_enemies)
            + (4 * self.num_closest_bombs)
            + (3 * self.num_closest_powerups)
        )

        # Using -inf to inf for positions/velocities, 0-1 for normalized hp, 0-max for counts
        # It's better to normalize these values later in _get_obs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self._game_activity: bs.GameActivity | None = None
        self._agent_player: bs.Player | None = None
        self._agent_spaz: PlayerSpaz | None = None
        self._bots: list[SpazBot] = []

        self._episode_step_count = 0
        self._max_episode_steps = (
            1000  # End episode after this many steps if not otherwise done
        )

        # For reward calculation
        self._prev_agent_hp = 0
        self._current_agent_hp = 0
        self._prev_agent_kills = 0  # Kills by agent's team
        self._current_agent_kills = 0
        self._prev_opponent_hp: dict[int, int] = {}
        self._current_opponent_hp: dict[int, int] = {}
        self._game_over_for_reward = False
        self._agent_won_for_reward = False

        self._last_move_lr = 0.0
        self._last_move_ud = 0.0

        self._game_ended_callback_set = False
        self._game_ended_flag = False  # Flag to signal game completion

    def _get_obs(self) -> np.ndarray:
        obs_list = []

        # Agent data
        if self._agent_spaz and self._agent_spaz.node:
            pos = self._agent_spaz.node.position
            vel = self._agent_spaz.node.velocity
            hp = (
                self._agent_spaz.hitpoints / self._agent_spaz.hitpoints_max
                if self._agent_spaz.hitpoints_max > 0
                else 0.0
            )
            bombs = (
                self._agent_spaz.bomb_count / 3.0
            )  # Normalize bomb count (assuming max 3)
            obs_list.extend([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], hp, bombs])
            self._current_agent_hp = self._agent_spaz.hitpoints
        else:
            obs_list.extend([0.0] * 8)  # Default if no spaz
            self._current_agent_hp = 0

        # Enemy data
        live_enemies = [bot for bot in self._bots if bot.is_alive() and bot.node]
        if self._agent_spaz and self._agent_spaz.node:
            agent_pos = bs.Vec3(self._agent_spaz.node.position)
            live_enemies.sort(
                key=lambda e: (bs.Vec3(e.node.position) - agent_pos).length()
            )

        self._current_opponent_hp.clear()
        for i in range(self.num_closest_enemies):
            if i < len(live_enemies):
                enemy = live_enemies[i]
                pos = enemy.node.position
                vel = enemy.node.velocity
                hp = (
                    enemy.hitpoints / enemy.hitpoints_max
                    if enemy.hitpoints_max > 0
                    else 0.0
                )
                obs_list.extend([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], hp])
                self._current_opponent_hp[enemy.node.node_id] = enemy.hitpoints
            else:
                obs_list.extend([0.0] * 7)  # Padding

        # Bomb data
        active_bombs = [
            n.getdelegate(bs.Bomb)
            for n in bs.getnodes()
            if n.getnodetype() == "bomb" and n.getdelegate(bs.Bomb)
        ]
        # Filter out None from getdelegate if bomb died mid-list comprehension
        active_bombs = [b for b in active_bombs if b and b.node]

        if self._agent_spaz and self._agent_spaz.node:
            agent_pos = bs.Vec3(self._agent_spaz.node.position)
            active_bombs.sort(
                key=lambda b: (bs.Vec3(b.node.position) - agent_pos).length()
            )

        for i in range(self.num_closest_bombs):
            if i < len(active_bombs):
                bomb = active_bombs[i]
                pos = bomb.node.position
                # Bomb timer is not directly accessible. We can approximate or use a fixed value.
                # For now, let's use a fixed value or 0 if it's an impact bomb.
                timer_normalized = 0.0 if bomb.bomb_type == "impact" else 1.0
                obs_list.extend([pos[0], pos[1], pos[2], timer_normalized])
            else:
                obs_list.extend([0.0] * 4)  # Padding

        # Powerup data
        active_powerups = [
            n.getdelegate(PowerupBox)
            for n in bs.getnodes()
            if n.getnodetype() == "prop" and n.getdelegate(PowerupBox)
        ]
        active_powerups = [
            p for p in active_powerups if p and p.node
        ]  # Filter out None

        if self._agent_spaz and self._agent_spaz.node:
            agent_pos = bs.Vec3(self._agent_spaz.node.position)
            active_powerups.sort(
                key=lambda p: (bs.Vec3(p.node.position) - agent_pos).length()
            )

        for i in range(self.num_closest_powerups):
            if i < len(active_powerups):
                powerup = active_powerups[i]
                pos = powerup.node.position
                obs_list.extend([pos[0], pos[1], pos.z])
            else:
                obs_list.extend([0.0] * 3)  # Padding

        return np.array(obs_list, dtype=np.float32)

    def _get_game_stats(self) -> None:
        if self._agent_player and self._agent_player.team:
            self._current_agent_kills = self._agent_player.team.score
        else:
            self._current_agent_kills = 0

        # Opponent HP is updated in _get_obs

    def _apply_action(self, action: int) -> None:
        if (
            not self._agent_spaz
            or not self._agent_spaz.is_alive()
            or not self._agent_spaz.node
        ):
            return

        # Reset continuous movement from last step unless a new move action is given
        current_move_lr = 0.0
        current_move_ud = 0.0

        if action == self.ACTION_MOVE_LEFT:
            current_move_lr = -1.0
        elif action == self.ACTION_MOVE_RIGHT:
            current_move_lr = 1.0
        elif action == self.ACTION_MOVE_UP:
            current_move_ud = (
                -1.0
            )  # Inverted Y for typical 2D top-down view if applicable
        elif action == self.ACTION_MOVE_DOWN:
            current_move_ud = 1.0
        elif action == self.ACTION_JUMP:
            self._agent_spaz.on_jump_press()
            self._agent_spaz.on_jump_release()  # Momentary
        elif action == self.ACTION_PUNCH:
            self._agent_spaz.on_punch_press()
            self._agent_spaz.on_punch_release()  # Momentary
        elif action == self.ACTION_BOMB:
            self._agent_spaz.on_bomb_press()
            self._agent_spaz.on_bomb_release()  # Momentary
        elif action == self.ACTION_PICKUP:
            self._agent_spaz.on_pickup_press()
            self._agent_spaz.on_pickup_release()  # Momentary
        elif action == self.ACTION_NOOP:
            # Keep previous movement if NOOP is chosen for action part
            current_move_lr = self._last_move_lr
            current_move_ud = self._last_move_ud
            pass

        # Apply movement (it's continuous until changed)
        self._agent_spaz.on_move_left_right(current_move_lr)
        self._agent_spaz.on_move_up_down(current_move_ud)
        self._last_move_lr = current_move_lr
        self._last_move_ud = current_move_ud

    def _game_has_ended_callback(self, game_results: bs.GameResults | None) -> None:
        """Callback from the game activity when it ends."""
        self._game_ended_flag = True
        self._game_over_for_reward = True
        if game_results and self._agent_player and self._agent_player.team:
            winning_team = game_results.winning_sessionteam
            if winning_team is self._agent_player.team.sessionteam:
                self._agent_won_for_reward = True
            else:
                self._agent_won_for_reward = False
        else:
            # If no results or no agent team, assume loss for reward calc
            self._agent_won_for_reward = False

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)  # Important for reproducibility if seed is used

        self._episode_step_count = 0
        self._game_ended_flag = False
        self._game_over_for_reward = False
        self._agent_won_for_reward = False
        self._last_move_lr = 0.0
        self._last_move_ud = 0.0

        # End any existing game activity for this environment
        if self._game_activity and not self._game_activity.expired:
            try:
                # Unset our callback first to avoid issues during forced end
                if hasattr(self._game_activity, "_rl_game_ended_callback"):
                    delattr(self._game_activity, "_rl_game_ended_callback")
                self._game_activity.end_game()  # Ask it to end gracefully
                # We might need to wait for it to truly end if it has delays
            except Exception as e:
                print(f"RL_ENV: Error ending previous game: {e}")

        self._game_activity = None
        self._agent_player = None
        self._agent_spaz = None
        self._bots.clear()

        # This is tricky. We need to launch a game and wait for it to be ready.
        # The training loop is external, so we can't easily use bs.apptimer here
        # without making reset asynchronous.
        # For now, let's assume the TrainingActivity handles launching the game
        # and then calls a method on the env to signal it's ready.
        # Or, the TrainingActivity passes the newly created activity to the env.

        if self._training_activity:
            self._training_activity.launch_new_game(self.map_name)

            # Wait for the game to be set up by TrainingActivity
            # This is a blocking wait, which is okay if reset() is not called too frequently
            # or if training is in a separate thread from the main game loop.
            # Given the setup, this will run in the main logic thread via the timer in TrainingActivity.
            start_wait = time.monotonic()
            while (
                not self._game_activity
                or not self._agent_spaz
                or not self._agent_spaz.is_alive()
            ):
                if time.monotonic() - start_wait > 10.0:  # 10 second timeout
                    raise RuntimeError(
                        "RL_ENV: Timeout waiting for game to start in reset()"
                    )
                # We need to let Ballistica's event loop run to process game setup.
                # This is a HACK. A proper way would be to make reset async or use callbacks.
                # If bs.run_for_ms exists and works as expected (processes events & renders):
                if hasattr(bs, "run_for_ms"):
                    bs.run_for_ms(10)  # Process for 10ms
                else:
                    # This is a fallback if run_for_ms is not available or doesn't work.
                    # It's not ideal as it relies on internal details.
                    if hasattr(bui.app, "_update_app_step"):
                        bui.app._update_app_step()  # type: ignore
                    time.sleep(0.01)  # Small sleep to prevent busy-waiting too hard

                # Re-check if training_activity has set them
                if not self._game_activity and hasattr(
                    self._training_activity, "current_game_activity"
                ):
                    self.set_game_activity(self._training_activity.current_game_activity)  # type: ignore
                if not self._agent_player and hasattr(
                    self._training_activity, "agent_player"
                ):
                    self.set_agent_player(self._training_activity.agent_player)  # type: ignore

        else:
            raise RuntimeError("RL_ENV: TrainingActivity reference not set.")

        # Set up a callback in the game activity to notify us when it truly ends
        if self._game_activity and not self._game_ended_callback_set:
            # Add a custom attribute to the game activity to hold our callback
            setattr(
                self._game_activity,
                "_rl_game_ended_callback",
                self._game_has_ended_callback,
            )
            self._game_ended_callback_set = True

        # Initial stats
        self._prev_agent_hp = self._agent_spaz.hitpoints if self._agent_spaz else 0
        self._current_agent_hp = self._prev_agent_hp
        self._prev_agent_kills = (
            self._agent_player.team.score
            if self._agent_player and self._agent_player.team
            else 0
        )
        self._current_agent_kills = self._prev_agent_kills
        self._prev_opponent_hp.clear()
        for bot in self._bots:
            if bot.node:
                self._prev_opponent_hp[bot.node.node_id] = bot.hitpoints

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if (
            not self._game_activity
            or not self._agent_spaz
            or not self._agent_spaz.is_alive()
        ):
            # If agent is dead or game ended unexpectedly, end episode
            # Return current (possibly stale) obs, 0 reward, and done=True
            return self._get_obs(), 0.0, True, False, {}

        self._prev_agent_hp = self._current_agent_hp  # From last step's _get_obs
        self._prev_agent_kills = (
            self._current_agent_kills
        )  # From last step's _get_game_stats
        self._prev_opponent_hp.clear()
        self._prev_opponent_hp.update(
            self._current_opponent_hp
        )  # From last step's _get_obs

        self._apply_action(int(action.item()))  # Assuming Discrete action space

        # Let the game run for a short period (e.g., a few frames)
        # This is where the rendering happens because the main loop is running.
        # The duration here controls how "fast" the RL agent perceives time.
        # This is a blocking call if bs.run_for_ms is not available or if we use time.sleep
        # For now, we assume the TrainingActivity's timer handles the "wait" part.
        # The step itself should be quick.
        # If bs.run_for_ms exists and works:
        # bs.run_for_ms(33) # Advance game by ~1/30th of a second

        # After action, update current stats for reward calculation
        self._get_game_stats()  # Updates self._current_agent_kills
        obs = (
            self._get_obs()
        )  # Updates self._current_agent_hp and self._current_opponent_hp

        reward = rl_reward.calculate_reward(
            self._game_activity,
            self._agent_spaz,
            self._prev_agent_hp,
            self._current_agent_hp,
            self._prev_agent_kills,
            self._current_agent_kills,
            self._prev_opponent_hp,
            self._current_opponent_hp,
            self._game_over_for_reward,  # This will be set by _game_has_ended_callback
            self._agent_won_for_reward,  # This will be set by _game_has_ended_callback
        )

        self._episode_step_count += 1

        done = False
        if not self._agent_spaz or not self._agent_spaz.is_alive():
            done = True
        if self._game_ended_flag:  # Set by the callback
            done = True

        truncated = False
        if self._episode_step_count >= self._max_episode_steps:
            truncated = True
            done = True  # Also end if truncated

        # Reset for next reward calculation after this step's reward is computed
        if done:
            self._game_over_for_reward = False
            self._agent_won_for_reward = False
            self._game_ended_callback_set = False  # Ready for next game

        return obs, reward, done, truncated, {}

    def render(self) -> None:
        # Ballistica handles its own rendering through its main loop.
        # If this method is called by SB3, we don't need to do anything extra
        # as long as the training loop allows the main game loop to run.
        pass

    def close(self) -> None:
        if self._game_activity and not self._game_activity.expired:
            try:
                if hasattr(self._game_activity, "_rl_game_ended_callback"):
                    delattr(self._game_activity, "_rl_game_ended_callback")
                self._game_activity.end(results=None, force=True)  # Force end
            except Exception as e:
                print(f"RL_ENV: Error closing game activity: {e}")
        self._game_activity = None
        self._agent_player = None
        self._agent_spaz = None
        self._bots.clear()

    # --- Helper methods for TrainingActivity to interact with the env ---
    def set_game_activity(self, activity: bs.GameActivity | None) -> None:
        self._game_activity = activity
        if activity:
            # Try to find bots. This might need to be more robust.
            if hasattr(activity, "_bots") and isinstance(
                getattr(activity, "_bots"), SpazBotSet
            ):
                bot_set = cast(SpazBotSet, getattr(activity, "_bots"))
                self._bots = (
                    bot_set.get_living_bots()
                )  # Or maintain a reference to the set
            else:  # Fallback: find all SpazBots in the activity
                self._bots = [
                    bs.Actor.getdelegate(SpazBot, n)  # type: ignore
                    for n in bs.getnodes()
                    if n.getnodetype() == "spaz" and bs.Actor.getdelegate(SpazBot, n)
                ]
                self._bots = [b for b in self._bots if b]  # Filter out Nones

    def set_agent_player(self, player: bs.Player | None) -> None:
        self._agent_player = player
        if player and player.actor:
            self._agent_spaz = cast(PlayerSpaz, player.actor)
        else:
            self._agent_spaz = None
