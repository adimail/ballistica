# Released under the MIT License. See LICENSE for details.
#
from __future__ import annotations

import logging
from typing import Any, override

import bascenev1 as bs
import bauiv1 as bui


class RLGameSession(bs.FreeForAllSession):
    """A FreeForAllSession tailored to launch a specific map and game type."""

    def __init__(self) -> None:
        app = bui.app
        assert app.classic is not None

        custom_game_config = getattr(app.classic, "custom_game_config", None)
        if custom_game_config is None or "map_name" not in custom_game_config:
            logging.error("RLGameSession started without custom_game_config.")
            map_name = "Courtyard"
        else:
            map_name = custom_game_config["map_name"]

        bui.app.config["Show Tutorial"] = False

        # Store original playlist settings to restore them later
        original_playlist_selection = app.config.get(self._playlist_selection_var)
        original_playlist_randomize = app.config.get(self._playlist_randomize_var)
        original_playlists_config = app.config.get(self._playlists_var)

        temp_playlist_name = "Custom_Death_Match"

        game_spec = {
            "type": "bascenev1lib.game.deathmatch.DeathMatchGame",
            "settings": {
                "map": map_name,
                "Time Limit": 200,
                "Kills to Win Per Player": 99999,  # Effectively infinite kill limit
                "Respawn Times": 1.0,
                "Epic Mode": False,
                "Allow Negative Scores": False,
            },
        }

        if self._playlists_var not in app.config:
            app.config[self._playlists_var] = {}

        app.config[self._playlists_var][temp_playlist_name] = [game_spec]
        app.config[self._playlist_selection_var] = temp_playlist_name
        app.config[self._playlist_randomize_var] = False

        try:
            super().__init__()
        finally:
            if original_playlist_selection is None:
                if self._playlist_selection_var in app.config:
                    del app.config[self._playlist_selection_var]
            else:
                app.config[self._playlist_selection_var] = original_playlist_selection

            if original_playlist_randomize is None:
                if self._playlist_randomize_var in app.config:
                    del app.config[self._playlist_randomize_var]
            else:
                app.config[self._playlist_randomize_var] = original_playlist_randomize

            if temp_playlist_name in app.config.get(self._playlists_var, {}):
                del app.config[self._playlists_var][temp_playlist_name]

            if original_playlists_config is None:
                if self._playlists_var in app.config:
                    del app.config[self._playlists_var]
            else:
                app.config[self._playlists_var] = original_playlists_config

            app.config.commit()

            if hasattr(app.classic, "custom_game_config"):
                delattr(app.classic, "custom_game_config")

    @override
    def on_activity_end(self, activity: bs.Activity, results: Any) -> None:
        # This method handles transitioning from the game to the score screen,
        # and then from the score screen to ending the session (which usually
        # means going back to the main menu or the next game in a series).
        # The existing logic here should work fine for your needs.
        # The score screen will show who won based on kills when the time limit is up.

        from bascenev1lib.activity.drawscore import DrawScoreScreenActivity
        from bascenev1lib.activity.freeforallvictory import (
            FreeForAllVictoryScoreScreenActivity,
        )

        if isinstance(activity, bs.GameActivity):
            winners = results.winnergroups
            if len(self.sessionplayers) > 1 and len(winners) < 2:
                next_activity_type = DrawScoreScreenActivity
            else:
                next_activity_type = FreeForAllVictoryScoreScreenActivity

            self.setactivity(bs.newactivity(next_activity_type, {"results": results}))

        elif isinstance(
            activity, (FreeForAllVictoryScoreScreenActivity, DrawScoreScreenActivity)
        ):
            self.end()
        else:
            super().on_activity_end(activity, results)
