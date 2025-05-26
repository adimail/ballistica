1.  **`babase`**: The core, foundational Python package. It handles:

    - The main `App` class, app state, and lifecycle.
    - Subsystem management (`AppSubsystem`, `AppComponentSubsystem`).
    - Configuration (`AppConfig`).
    - Intents and AppModes for high-level state management.
    - Core utilities: logging, errors, general Python helpers, math, text formatting.
    - Interfacing with the native C++ layer (`_babase`) for many low-level operations (timing, input, graphics, sound, etc.).
    - Localization (`Lstr`, `LanguageSubsystem`, `LocaleSubsystem`).
    - Account management (v2 via `_accountv2.py`, login adapters).
    - Metadata scanning for discovering plugins and other components.
    - Plugin management.
    - Asyncio integration.
    - Developer console.
    - Workspace and asset management (though `_assetmanager.py` seems a bit debug-heavy).

2.  **`baclassic`**: This package contains components specific to the "classic" BombSquad experience.

    - `ClassicAppMode` and `ClassicAppSubsystem`.
    - Legacy V1 account handling (`_accountv1.py`).
    - Achievements, ads, music, store, tips, and tournament logic specific to the classic game.
    - Input device mapping and server mode controller.

3.  **`baplus`**: This seems to be for "Plus" features, likely involving more advanced or possibly closed-source backend interactions.

    - `PlusAppSubsystem`.
    - `CloudSubsystem` for master-server communication.

4.  **`bascenev1`**: This is the gameplay-centric API.

    - Defines core gameplay entities: `Session`, `Activity`, `Actor`, `Player`, `Team`.
    - Handles game logic flow: `GameActivity`, `CoopGameActivity`, `TeamGameActivity`.
    - Manages game elements: `Map`, `Level`, `Campaign`, `Bomb`, `Flag`, `PowerupBox`, `Spaz`.
    - Defines message types for actor communication (e.g., `DieMessage`, `HitMessage`).
    - Includes a dependency system for assets and components.
    - Provides default playlists and score/stats management.

5.  **`bascenev1lib`**: This library contains concrete implementations of actors, games, and other gameplay elements built on top of `bascenev1`.

    - Specific game modes like `DeathMatchGame`, `CaptureTheFlagGame`, `NinjaFightGame`, `OnslaughtGame`, `RaceGame`, `KeepAwayGame`, `FootballGame`, `TargetPracticeGame`, `EasterEggHuntGame`.
    - Actor implementations like `Spaz`, `PlayerSpaz`, `SpazBot`, `Bomb`, `Flag`, `PowerupBox`, `Scoreboard`, `Text`, `Image`, `OnScreenTimer`, `OnScreenCountdown`, `ControlsGuide`, `TipsText`, `ZoomText`.
    - The main menu activity and tutorial.

6.  **`bauiv1`**: This is the user interface API (version 1).

    - Defines UI core types: `Window`, `MainWindow`, `MainWindowState`.
    - Provides widget creation functions (buttonwidget, textwidget, etc.) that wrap native UI elements.
    - Manages the root UI and UI lifecycle events.

7.  **`bauiv1lib`**: This library contains concrete UI window implementations built on `bauiv1`.

    - Windows for settings (graphics, controls, audio, advanced, plugins, dev tools, etc.).
    - Windows for gameplay setup (co-op browser, playlist browser, etc.).
    - Windows for account management, store, achievements, party, etc.
    - Utility UI like popups, color pickers, file selectors.

8.  **`batemplatefs`**: A template feature set, likely for developers to use as a starting point for their own feature sets.

9.  **`baenv.py`**: Handles the initial Python environment setup, including paths, logging, and configuration loading, before the main Ballistica modules are imported.
