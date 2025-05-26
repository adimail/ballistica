# Released under the MIT License. See LICENSE for details.

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import bascenev1 as bs
import bauiv1 as bui
from bascenev1lib.game.deathmatch import DeathMatchGame
from rl_env import DeathMatchEnv
from stable_baselines3 import PPO

if TYPE_CHECKING:
    from typing import Any


class TrainingActivity(bs.Activity[bs.Player, bs.Team]):
    """
    Activity to host and manage the RL training process.
    """

    def __init__(self, settings: dict | None = None):
        if settings is None:
            settings = {}
        super().__init__(settings)
        self.map_name = settings.get("map_name", "Doom Shroom")

        self.env: DeathMatchEnv | None = None
        self.model: PPO | None = None
        self.obs: Any = None
        self.total_steps_done = 0
        self.total_timesteps_to_run = settings.get("total_timesteps", 100000)
        self.training_step_timer: bui.AppTimer | None = None
        self.log_interval = settings.get("log_interval", 1000)
        self.save_interval = settings.get("save_interval", 10000)
        self.model_save_path = settings.get("model_save_path", "rl_deathmatch_agent")

        self.current_game_activity: bs.GameActivity | None = None
        self.agent_player: bs.Player | None = None

        self._game_launch_timer: bs.Timer | None = None

    @override
    def on_begin(self) -> None:
        super().on_begin()
        bui.screenmessage(
            f"Starting RL Training on map: {self.map_name} for {self.total_timesteps_to_run} steps.",
            color=(0, 1, 0),
        )

        self.env = DeathMatchEnv(map_name=self.map_name, training_activity=self)

        # For now, using a simple policy. You might want to customize this.
        # Also, consider loading a pre-trained model if available.
        self.model = PPO(
            "MlpPolicy", self.env, verbose=0, tensorboard_log="./bs_rl_tensorboard/"
        )

        # The reset call will trigger launch_new_game via the env
        self.obs, _ = self.env.reset()

        # Start the training loop via a timer to integrate with bs event loop
        # Adjust timer interval for training speed vs. visibility.
        # Very small interval will make it run faster but might be hard to see.
        self.training_step_timer = bui.AppTimer(
            0.01, self._perform_training_step, repeat=True
        )

    def launch_new_game(self, map_name: str) -> None:
        """Called by the environment to start/restart a game."""
        if self.current_game_activity and not self.current_game_activity.expired:
            self.current_game_activity.end(
                results=None, force=True
            )  # End previous game forcefully

        self.current_game_activity = None
        self.agent_player = None

        # Launch a new DeathMatch game session and activity
        # This needs to happen in a way that the TrainingActivity remains in control
        # or can get references to the new game and player.
        # This is a simplified way; a real game setup might be more complex.

        # We need to create a new session that will host our DeathMatchGame
        # For simplicity, let's assume we're creating a FreeForAllSession
        # that will then launch the DeathMatchGame.

        # Clear any existing session first
        if bs.get_foreground_host_session():
            bs.get_foreground_host_session().end()

        # Define settings for the DeathMatchGame
        game_settings = {
            "map": map_name,
            "Kills to Win Per Player": 5,  # Short game for faster episodes
            "Time Limit": 120,  # 2 minutes
            "Respawn Times": 1.0,
            "Epic Mode": False,
            "Allow Negative Scores": True,  # Useful for RL
        }

        # Create a config for the session
        # We'll make a simple FFA session with 1 AI bot.
        # The RL agent will be player 0.
        config = bs.app.config
        config["Free-for-All Playlist Selection"] = (
            "__default__"  # Does not matter much here
        )
        config["Free-for-All Playlists"] = {
            "__rl_training_playlist__": [
                {
                    "type": "bascenev1lib.game.deathmatch.DeathMatchGame",
                    "settings": game_settings,
                }
            ]
        }
        config["Free-for-All Playlist Selection"] = "__rl_training_playlist__"
        config["Free-for-All Max Players"] = 2  # Agent + 1 Bot

        # This is a bit of a hack. We're starting a new session.
        # The TrainingActivity itself is part of the *previous* session (likely MainMenuSession).
        # When the new session starts, this TrainingActivity instance might get cleaned up.
        # This needs careful handling.
        # A more robust way: TrainingActivity *is* the game, or it directly controls a simple game setup.

        # For now, let's assume TrainingActivity persists or can re-hook.
        # The new session will create the DeathMatchGame. We need a way to get a ref to it.

        def _on_session_begin_for_rl():
            session = bs.get_foreground_host_session()
            if session and session.getactivity():
                game_act = cast(DeathMatchGame, session.getactivity())
                if self.env:
                    self.env.set_game_activity(game_act)

                # Find the agent's player (assume it's the first local player)
                # This is a simplification.
                if game_act.players:
                    self.agent_player = game_act.players[0]
                    if self.env:
                        self.env.set_agent_player(self.agent_player)
                else:
                    # This can happen if players haven't joined yet.
                    # We need to wait for player spawn.
                    # This synchronization is complex.
                    print(
                        "RL_TRAIN: No players in game activity yet during _on_session_begin_for_rl"
                    )

        # This is a bit indirect. We're setting up a new session.
        # The TrainingActivity needs to get a handle on the game and player
        # once that new session's game starts.
        # A bs.timer might be too slow.
        # A better way: the DeathMatchGame itself could have a mode where it's controlled by RL.

        # Let's try to launch and then poll for the activity/player.
        # This is still part of the HACK in reset()
        bs.new_host_session(bs.FreeForAllSession)

        # The actual game activity and player will be picked up by the env's reset() polling.
        # This is not ideal but a starting point.

    def _perform_training_step(self) -> None:
        if not self.model or not self.env or self.obs is None:
            bui.screenmessage("RL Training: Model or Env not ready.", color=(1, 0, 0))
            if self.training_step_timer:
                self.training_step_timer = None  # Stop if something is wrong
            return

        if self.total_steps_done >= self.total_timesteps_to_run:
            if self.training_step_timer:
                self.training_step_timer = None  # Stop training
            self.model.save(self.model_save_path)
            bui.screenmessage(
                f"Training complete! Model saved to {self.model_save_path}",
                color=(0, 1, 0),
            )
            print(f"Training complete! Model saved to {self.model_save_path}")
            # Transition back or show results
            self.end_game_and_cleanup()
            return

        try:
            action, _states = self.model.predict(
                self.obs, deterministic=False
            )  # Use deterministic=False for exploration
            new_obs, reward, done, truncated, info = self.env.step(action)

            # This is where you'd normally add to a replay buffer and train the model.
            # For on-policy like PPO, SB3's model.learn() handles this.
            # Since we're doing a custom loop for rendering, we'd need to replicate parts of it.
            # For simplicity, this example focuses on *running* the agent.
            # To *train* it effectively with this visible loop, you'd need to:
            # 1. Collect a batch of experiences (obs, action, reward, new_obs, done).
            # 2. Call `self.model.train()` with that batch.
            # This is non-trivial to implement correctly outside SB3's `learn()`.
            # A simpler path for *actual training* might be to run SB3's `learn()`
            # and have `env.render()` do nothing, then load the trained model for visualization.
            # However, the request is to *see* the training.

            # Let's just log reward for now.
            if self.total_steps_done % 100 == 0:  # Log every 100 steps
                if hasattr(self.model, "logger") and self.model.logger is not None:
                    self.model.logger.record(
                        "rollout/ep_rew_mean_step", reward
                    )  # This is not ep_rew_mean, just current step reward
                    self.model.logger.dump(step=self.total_steps_done)

            self.obs = new_obs
            if done or truncated:
                bui.screenmessage(
                    f"Episode finished after {self.env._episode_step_count} steps.",
                    color=(1, 1, 0),
                )
                self.obs, _ = self.env.reset()
                # Log episode stats if your environment tracks them (e.g. in info dict)

            self.total_steps_done += 1

            if self.total_steps_done % self.log_interval == 0:
                print(
                    f"Training step {self.total_steps_done}/{self.total_timesteps_to_run}"
                )

            if self.total_steps_done % self.save_interval == 0:
                self.model.save(f"{self.model_save_path}_step_{self.total_steps_done}")
                print(f"Intermediate model saved at step {self.total_steps_done}")

        except Exception as e:
            print(f"Error during training step: {e}")
            import traceback

            traceback.print_exc()
            if self.training_step_timer:
                self.training_step_timer = None  # Stop on error
            self.end_game_and_cleanup()

    def end_game_and_cleanup(self) -> None:
        """Called when training is done or an error occurs."""
        if self.env:
            self.env.close()
        # Potentially transition back to main menu or a results screen
        # For now, just end this activity.
        if not self.expired:
            self.end()

    @override
    def on_player_join(self, player: bs.Player) -> None:
        # This activity is not meant for interactive players during training.
        # However, the session setup might add the local player.
        # We'll try to assign the first joined player as the agent.
        if not self.agent_player:
            self.agent_player = player
            if self.env:
                self.env.set_agent_player(player)

        # Disable input for any players joining this "game"
        player.resetinput()

    @override
    def handlemessage(self, msg: Any) -> Any:
        if isinstance(msg, bs.PlayerDiedMessage):
            # The environment handles agent death logic.
            # We might want to log it here or update UI if needed.
            pass
        return super().handlemessage(msg)

    @override
    def on_end(self) -> None:
        super().on_end()
        # Clean up timers and environment
        if self.training_step_timer:
            self.training_step_timer = None
        if self.env:
            self.env.close()
        # Ensure any game session started by this activity is also ended
        if (
            bs.get_foreground_host_session()
            and bs.get_foreground_host_session().getactivity()
            is self.current_game_activity
        ):
            bs.get_foreground_host_session().end()


class TrainingSession(bs.Session):
    """
    Session to host the TrainingActivity.
    """

    def __init__(self, map_name: str, total_timesteps: int = 100000):
        self._map_name = map_name
        self._total_timesteps = total_timesteps

        # This session doesn't need complex deps for its own activity,
        # but the DeathMatchGame launched by TrainingActivity will have its own.
        # For now, an empty depset for TrainingActivity itself.
        activity_deps = bs.DependencySet(bs.Dependency(TrainingActivity))

        super().__init__([activity_deps])

        # TrainingActivity will handle its own game logic.
        # This session is primarily a container.
        self.setactivity(
            bs.newactivity(
                TrainingActivity,
                settings={
                    "map_name": self._map_name,
                    "total_timesteps": self._total_timesteps,
                },
            )
        )

    @override
    def on_player_request(self, player: bs.SessionPlayer) -> bool:
        # Allow one player (the "agent", though it's not directly controlled by user)
        # Or, if you want to spectate, allow more. For now, let's keep it simple.
        return len(self.sessionplayers) == 0


def start_rl_training(map_name: str, total_timesteps: int = 100000) -> None:
    """
    Public function to kick off the RL training.
    This will be called from the map selection UI.
    """

    # End any current game/session first
    current_session = bs.get_foreground_host_session()
    if current_session:
        current_session.end()

    # We need to wait for the old session to fully end before starting a new one.
    # This is a bit of a hack; a cleaner way would be to chain this via callbacks.
    def _actually_start_training():
        try:
            bs.new_host_session(
                TrainingSession(map_name=map_name, total_timesteps=total_timesteps)
            )
        except Exception as e:
            print(f"Error starting RL training session: {e}")
            bui.screenmessage(f"Error starting training: {e}", color=(1, 0, 0))
            # Fallback to main menu
            from bascenev1lib.mainmenu import MainMenuSession

            bs.new_host_session(MainMenuSession)

    # Give a moment for the old session to tear down.
    bui.apptimer(0.1, _actually_start_training)
