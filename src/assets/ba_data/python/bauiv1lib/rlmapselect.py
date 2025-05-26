# Released under the MIT License. See LICENSE for details.


from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import bascenev1 as bs
import bauiv1 as bui
import rl_train

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple


class RLMapSelectWindow(bui.MainWindow):
    """Window for selecting a map to run RL training on."""

    def __init__(
        self, transition: str = "in_right", origin_widget: bui.Widget | None = None
    ):
        app = bui.app
        assert app.classic is not None
        uiscale = app.ui_v1.uiscale

        self._width = 600
        self._height = 400 if uiscale is bui.UIScale.SMALL else 500

        self._cancel_button = bui.buttonwidget(
            parent=self._root_widget,
            position=(40, self._height - 60),
            size=(120, 50),
            scale=0.8,
            label=bui.Lstr(resource="cancelText"),
            on_activate_call=self.main_window_back,
        )
        bui.containerwidget(edit=self._root_widget, cancel_button=self._cancel_button)

        bui.textwidget(
            parent=self._root_widget,
            position=(self._width * 0.5, self._height - 35),
            size=(0, 0),
            text="Select Map for RL Training",  # TODO: Localize
            color=app.ui_v1.title_color,
            h_align="center",
            v_align="center",
            maxwidth=self._width * 0.8,
        )

        self._scroll_width = self._width - 80
        self._scroll_height = self._height - 120
        self._scrollwidget = bui.scrollwidget(
            parent=self._root_widget,
            position=(40, 50),
            size=(self._scroll_width, self._scroll_height),
        )
        bui.containerwidget(edit=self._root_widget, selected_child=self._scrollwidget)

        self._columnwidget = bui.columnwidget(
            parent=self._scrollwidget, border=2, margin=0
        )

        self._available_maps = app.classic.getmaps("melee")  # For DeathMatch
        for i, map_name in enumerate(self._available_maps):
            map_display_name = bs.get_map_display_string(map_name)
            btn = bui.buttonwidget(
                parent=self._columnwidget,
                size=(self._scroll_width - 40, 50),
                label=map_display_name,
                autoselect=True,
                on_activate_call=bui.Call(self._select_map, map_name),
            )
            if i == 0:
                bui.widget(edit=btn, up_widget=self._cancel_button)

    def _select_map(self, map_name: str) -> None:
        if not self.main_window_has_control():
            return

        # Close ourself
        self.main_window_back()

        # Kick off training
        # You might want to add a settings window here for total_timesteps, etc.
        rl_train.start_rl_training(
            map_name=map_name, total_timesteps=200000
        )  # Example timesteps

    @override
    def get_main_window_state(self) -> bui.MainWindowState:
        cls = type(self)
        return bui.BasicMainWindowState(
            create_call=lambda transition, origin_widget: cls(
                origin_widget=origin_widget
            )
        )
