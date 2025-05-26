# Released under the MIT License. See LICENSE for details.


from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import bascenev1 as bs
import bauiv1 as bui

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple


class RLMapSelectWindow(bui.MainWindow):
    """Window for selecting a map to start a game on."""

    def __init__(
        self, transition: str = "in_right", origin_widget: bui.Widget | None = None
    ):
        app = bui.app
        assert app.classic is not None
        uiscale = app.ui_v1.uiscale

        self._width = 600
        self._height = 400 if uiscale is bui.UIScale.SMALL else 500

        # Define the root widget FIRST.
        # This is the main container for your window's UI elements.
        root_widget = bui.containerwidget(
            size=(self._width, self._height),
            scale=(
                2.0
                if uiscale is bui.UIScale.SMALL
                else 1.5 if uiscale is bui.UIScale.MEDIUM else 1.0
            ),
            stack_offset=(0, -10 if uiscale is bui.UIScale.SMALL else 0),
        )

        # Call super().__init__ and pass the root_widget.
        # Also pass other relevant args like transition and origin_widget.
        super().__init__(
            root_widget=root_widget, transition=transition, origin_widget=origin_widget
        )

        # Now self._root_widget is available and refers to the root_widget
        # we just created and passed to the superclass.
        self._cancel_button = bui.buttonwidget(
            parent=self._root_widget,
            position=(40, self._height - 60),
            size=(120, 50),
            scale=0.8,
            label=bui.Lstr(resource="cancelText"),
            on_activate_call=self.main_window_back,
            autoselect=True,
        )
        bui.containerwidget(edit=self._root_widget, cancel_button=self._cancel_button)

        bui.textwidget(
            parent=self._root_widget,
            position=(self._width * 0.5, self._height - 35),
            size=(0, 0),
            text="Select Map for Game",  # Updated text
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
            autoselect=True,  # Make the scroll area selectable
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
            # Ensure proper navigation from cancel button to the list
            # And from list back to cancel button if back_button is not present (small UI)
            if self._cancel_button:  # Assuming self._back_button is set if not small UI
                bui.widget(
                    edit=self._cancel_button,
                    down_widget=(
                        btn if i == 0 else self._cancel_button.get_selected_child()
                    ),
                )
            else:  # For small UI scale, back button is special
                bui.widget(
                    edit=self._cancel_button,
                    down_widget=(
                        btn if i == 0 else bui.get_special_widget("back_button")
                    ),
                )

    def _select_map(self, map_name: str) -> None:
        if not self.main_window_has_control():
            return

        # Store the selected map for our custom session to pick up.
        if bui.app.classic is None:
            logging.error(
                "RLMapSelectWindow: Classic app subsystem not found. Cannot store map selection."
            )
            bui.screenmessage("Error: Game components missing.", color=(1, 0, 0))
            bui.getsound("error").play()
            return

        # Use a unique attribute name to avoid conflicts
        if not hasattr(bui.app.classic, "custom_game_config"):
            bui.app.classic.custom_game_config = {}
        bui.app.classic.custom_game_config["map_name"] = map_name
        bui.app.classic.custom_game_config["game_type"] = (
            "DeathMatch"  # You can make this more generic
        )

        # Close ourself
        self.main_window_back()

        # Fade screen and then launch the game session
        bui.fade_screen(False, endcall=self._launch_game_session)

    def _launch_game_session(self) -> None:
        # Import RLGameSession here to avoid circular dependencies at module load time
        # if RLGameSession is in another file.
        from bauiv1lib.rl_game_session import RLGameSession  # Create this file next

        try:
            bs.new_host_session(RLGameSession)
        except Exception:
            logging.exception("Error creating RLGameSession.")
            # Fallback to main menu
            from bascenev1lib.mainmenu import MainMenuSession

            bs.new_host_session(MainMenuSession)

    @override
    def get_main_window_state(self) -> bui.MainWindowState:
        cls = type(self)
        return bui.BasicMainWindowState(
            create_call=lambda transition, origin_widget: cls(
                transition=transition,  # Pass transition here
                origin_widget=origin_widget,
            )
        )
