#!/usr/bin/env python3
"""
BombSquad Remote Controller

UDP-based remote controller for BombSquad game using pygame.
Supports V2 protocol.
"""

import logging
import socket
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set, Tuple

import pygame

# Configuration
BOMBSQUAD_IP = "127.0.0.1"
BOMBSQUAD_PORT = 43210
PLAYER_NAME = "PyRemote"

# Protocol constants
REMOTE_MSG_ID_REQUEST = 2
REMOTE_MSG_ID_RESPONSE = 3
REMOTE_MSG_DISCONNECT = 4
REMOTE_MSG_STATE = 5
REMOTE_MSG_STATE_ACK = 6
REMOTE_MSG_STATE2 = 10

# Button bitmasks for V2 state protocol
BUTTON_MENU = 1
BUTTON_JUMP = 1 << 1
BUTTON_PUNCH = 1 << 2
BUTTON_PICK_UP = 1 << 3
BUTTON_BOMB = 1 << 4
BUTTON_RUN = 1 << 5

# Protocol versions
PROTOCOL_VERSION_OLD = 121
PROTOCOL_VERSION_NEW_REQUEST = 50
PROTOCOL_VERSION_V2_ACK = 100

# Connection parameters
JOIN_REQUEST_INTERVAL = 2.0
CONNECTION_TIMEOUT = 10.0
STATE_PACKET_RATE = 30  # FPS
MAX_RECONNECT_ATTEMPTS = 5

# Display settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
CONNECT_BUTTON_WIDTH = 150
CONNECT_BUTTON_HEIGHT = 40


class UIState(Enum):
    """UI connection states"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class ConnectionState:
    """Manages connection state and statistics"""

    ui_state: UIState = UIState.DISCONNECTED
    assigned_controller_id: int = -1
    using_protocol_v2: bool = True
    next_state_packet_id: int = 0
    last_ack_received_id: int = -1
    last_join_request_time: float = 0
    connection_start_time: float = 0
    reconnect_attempts: int = 0


@dataclass
class InputState:
    """Manages input state"""

    pressed_pygame_keys_display: Set[str]
    current_button_mask: int = 0
    dpad_h_float: float = 0.0
    dpad_v_float: float = 0.0

    def __post_init__(self):
        if not hasattr(self, "pressed_pygame_keys_display"):
            self.pressed_pygame_keys_display = set()


class BombSquadRemoteController:
    """Main controller class for BombSquad remote connection"""

    def __init__(self):
        self.key_bitmask_map = {
            pygame.K_k: ("K (Jump)", BUTTON_JUMP),
            pygame.K_j: ("J (Punch)", BUTTON_PUNCH),
            pygame.K_o: ("O (Bomb)", BUTTON_BOMB),
            pygame.K_i: ("I (Pickup)", BUTTON_PICK_UP),
            pygame.K_RETURN: ("ENTER (Menu)", BUTTON_MENU),
            pygame.K_LSHIFT: ("LSHIFT (Run)", BUTTON_RUN),
        }

        self.connection = ConnectionState()
        self.input_state = InputState(pressed_pygame_keys_display=set())
        self.sock: Optional[socket.socket] = None
        self._setup_logging()
        self._initialize_socket()

    def _setup_logging(self) -> None:
        """Configure logging for debugging"""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_socket(self) -> bool:
        """Initialize UDP socket with proper error handling"""
        try:
            if self.sock:
                self.sock.close()

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.logger.info("Socket initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize socket: {e}")
            self.connection.ui_state = UIState.FAILED
            return False

    def send_join_request_packet(self) -> bool:
        """Send join request packet to BombSquad server"""
        if self.sock is None:
            return False

        try:
            request_id = int(time.monotonic() * 1000) % 10000
            name_bytes = PLAYER_NAME.encode("utf-8")[:99]

            message_parts = [
                REMOTE_MSG_ID_REQUEST,
                PROTOCOL_VERSION_OLD,
                request_id & 0xFF,
                (request_id >> 8) & 0xFF,
                PROTOCOL_VERSION_NEW_REQUEST,
            ]
            message = bytes(message_parts) + name_bytes

            self.sock.sendto(message, (BOMBSQUAD_IP, BOMBSQUAD_PORT))
            self.connection.last_join_request_time = time.time()
            self.logger.info("Join request sent")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send join request: {e}")
            return False

    def send_control_state_packet(self) -> bool:
        """Send control state packet using appropriate protocol version"""
        if (
            self.connection.assigned_controller_id == -1
            or self.connection.ui_state != UIState.CONNECTED
            or not self.sock
        ):
            return False

        try:
            if self.connection.using_protocol_v2:
                return self._send_v2_state_packet()
            else:
                return self._send_v1_state_packet()

        except Exception as e:
            self.logger.error(f"Failed to send control state: {e}")
            return False

    def _send_v2_state_packet(self) -> bool:
        """Send V2 protocol state packet"""
        dpad_h_byte = self._float_to_byte(self.input_state.dpad_h_float)
        dpad_v_byte = self._float_to_byte(self.input_state.dpad_v_float)

        message = bytes(
            [
                REMOTE_MSG_STATE2,
                self.connection.assigned_controller_id,
                1,  # Number of states in packet
                self.connection.next_state_packet_id,
                self.input_state.current_button_mask & 0xFF,
                dpad_h_byte,
                dpad_v_byte,
            ]
        )

        if self.sock:
            self.sock.sendto(message, (BOMBSQUAD_IP, BOMBSQUAD_PORT))
            self.connection.next_state_packet_id = (
                self.connection.next_state_packet_id + 1
            ) % 256

            return True

        return False

    def _send_v1_state_packet(self) -> bool:
        """Send V1 protocol state packet"""
        dpad_h_byte = self._float_to_byte(self.input_state.dpad_h_float)
        dpad_v_byte = self._float_to_byte(self.input_state.dpad_v_float)

        message = bytes(
            [
                REMOTE_MSG_STATE,
                self.connection.assigned_controller_id,
                dpad_h_byte,
                dpad_v_byte,
                self.input_state.current_button_mask & 0xFF,
            ]
        )

        if self.sock:
            self.sock.sendto(message, (BOMBSQUAD_IP, BOMBSQUAD_PORT))
            return True
        return False

    def _float_to_byte(self, value: float) -> int:
        """Convert float (-1.0 to 1.0) to byte (0-255)"""
        byte_value = int(127.5 + (value * 127.5))
        return max(0, min(255, byte_value))

    def send_disconnect_packet(self) -> bool:
        """Send disconnect packet to server"""
        if not self.sock or self.connection.assigned_controller_id == -1:
            return False

        try:
            message = bytes(
                [REMOTE_MSG_DISCONNECT, self.connection.assigned_controller_id]
            )
            self.sock.sendto(message, (BOMBSQUAD_IP, BOMBSQUAD_PORT))
            self.logger.info("Disconnect packet sent")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send disconnect: {e}")
            return False

    def process_incoming_packets(self) -> None:
        """Process all available incoming packets"""
        if not self.sock:
            return

        try:
            while True:
                try:
                    data, _ = self.sock.recvfrom(1024)
                    if not data:
                        break

                    self._handle_packet(data)

                except BlockingIOError:
                    break

        except Exception as e:
            self.logger.error(f"Error processing packets: {e}")

    def _handle_packet(self, data: bytes) -> None:
        """Handle individual received packet"""
        if len(data) == 0:
            return

        packet_type = data[0]

        if packet_type == REMOTE_MSG_ID_RESPONSE:
            self._handle_join_response(data)
        elif packet_type == REMOTE_MSG_STATE_ACK:
            self._handle_state_ack(data)
        elif packet_type == REMOTE_MSG_DISCONNECT:
            self._handle_disconnect()

    def _handle_join_response(self, data: bytes) -> None:
        """Handle join response packet"""
        if len(data) < 3 or self.connection.ui_state != UIState.CONNECTING:
            return

        self.connection.assigned_controller_id = data[1]
        server_protocol_ack = data[2]
        self.connection.using_protocol_v2 = (
            server_protocol_ack == PROTOCOL_VERSION_V2_ACK
        )

        protocol_version = "V2" if self.connection.using_protocol_v2 else "V1"
        self.logger.info(
            f"Connected! Controller ID: {self.connection.assigned_controller_id}, Protocol: {protocol_version}"
        )

        self.connection.ui_state = UIState.CONNECTED
        self.connection.reconnect_attempts = 0

    def _handle_state_ack(self, data: bytes) -> None:
        """Handle state acknowledgment packet"""
        if len(data) >= 2 and self.connection.ui_state == UIState.CONNECTED:
            self.connection.last_ack_received_id = data[1]

    def _handle_disconnect(self) -> None:
        """Handle disconnect packet from server"""
        self.logger.info("Received disconnect from server")
        self.connection.ui_state = UIState.DISCONNECTED
        self.connection.assigned_controller_id = -1

    def attempt_connect(self) -> None:
        """Attempt to connect to BombSquad server"""
        if not self._initialize_socket():
            return

        self._reset_connection_state()
        self.connection.ui_state = UIState.CONNECTING
        self.connection.connection_start_time = time.time()
        self.send_join_request_packet()

    def _reset_connection_state(self) -> None:
        """Reset connection state for new connection attempt"""
        self.connection.assigned_controller_id = -1
        self.connection.next_state_packet_id = 0
        self.connection.last_ack_received_id = -1

    def handle_connection_timeout(self) -> None:
        """Handle connection timeout and retry logic"""
        if self.connection.ui_state != UIState.CONNECTING:
            return

        current_time = time.time()

        if current_time - self.connection.connection_start_time > CONNECTION_TIMEOUT:
            self.connection.reconnect_attempts += 1

            if self.connection.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                self.logger.error("Max reconnection attempts reached")
                self.connection.ui_state = UIState.FAILED
            else:
                self.logger.warning(
                    f"Connection timeout, retry {self.connection.reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}"
                )
                self.attempt_connect()

        elif (
            current_time - self.connection.last_join_request_time
            > JOIN_REQUEST_INTERVAL
        ):
            self.send_join_request_packet()

    def handle_key_event(self, event: pygame.event.Event) -> None:
        """Handle pygame key events"""
        if self.connection.ui_state not in [UIState.CONNECTED, UIState.CONNECTING]:
            return

        if event.type == pygame.KEYDOWN:
            self._handle_key_down(event.key)
        elif event.type == pygame.KEYUP:
            self._handle_key_up(event.key)

    def _handle_key_down(self, key: int) -> None:
        """Handle key press events"""
        if key in self.key_bitmask_map:
            display_name, mask = self.key_bitmask_map[key]
            self.input_state.current_button_mask |= mask
            self.input_state.pressed_pygame_keys_display.add(display_name)

        self._update_dpad_from_keys()

    def _handle_key_up(self, key: int) -> None:
        """Handle key release events"""
        if key in self.key_bitmask_map:
            display_name, mask = self.key_bitmask_map[key]
            self.input_state.current_button_mask &= ~mask
            self.input_state.pressed_pygame_keys_display.discard(display_name)

        self._update_dpad_from_keys()

    def _update_dpad_from_keys(self) -> None:
        """Update D-pad state based on currently pressed keys"""
        keys = pygame.key.get_pressed()

        self.input_state.dpad_h_float = 0.0
        if keys[pygame.K_a]:
            self.input_state.dpad_h_float -= 1.0
        if keys[pygame.K_d]:
            self.input_state.dpad_h_float += 1.0

        self.input_state.dpad_v_float = 0.0
        if keys[pygame.K_w]:
            self.input_state.dpad_v_float -= 1.0
        if keys[pygame.K_s]:
            self.input_state.dpad_v_float += 1.0

    def get_status_info(self) -> Tuple[str, Tuple[int, int, int]]:
        """Get current status text and color"""
        if self.connection.ui_state == UIState.DISCONNECTED:
            return "Status: Disconnected", (200, 100, 100)
        elif self.connection.ui_state == UIState.CONNECTING:
            return "Status: Connecting...", (200, 200, 100)
        elif self.connection.ui_state == UIState.CONNECTED:
            return (
                f"Status: Connected (ID: {self.connection.assigned_controller_id})",
                (100, 200, 100),
            )
        elif self.connection.ui_state == UIState.FAILED:
            return "Status: Connection Failed", (200, 50, 50)
        else:
            return "Status: Unknown", (150, 150, 150)

    def get_active_controls(self) -> list:
        """Get list of currently active controls"""
        active_controls = []

        if self.input_state.dpad_v_float < 0:
            active_controls.append("Up")
        if self.input_state.dpad_v_float > 0:
            active_controls.append("Down")
        if self.input_state.dpad_h_float < 0:
            active_controls.append("Left")
        if self.input_state.dpad_h_float > 0:
            active_controls.append("Right")

        for key_name in self.input_state.pressed_pygame_keys_display:
            button_name = key_name.split("(")[1].split(")")[0].strip()
            active_controls.append(button_name)

        return active_controls

    def cleanup(self) -> None:
        """Clean up resources and send final neutral state"""
        if self.connection.assigned_controller_id != -1:
            self.logger.info("Sending neutral state on exit")
            self.input_state.current_button_mask = 0
            self.input_state.dpad_h_float = 0.0
            self.input_state.dpad_v_float = 0.0
            self.send_control_state_packet()
            time.sleep(0.1)
            self.send_disconnect_packet()

        if self.sock:
            self.sock.close()
            self.sock = None

        self.logger.info("Controller cleanup completed")


def main():
    """Entry point"""
    controller = BombSquadRemoteController()

    try:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"BombSquad PyRemote: {PLAYER_NAME}")

        font_small = pygame.font.Font(None, 20)
        font_large = pygame.font.Font(None, 28)
        clock = pygame.time.Clock()

        connect_button_rect = pygame.Rect(
            SCREEN_WIDTH // 2 - CONNECT_BUTTON_WIDTH // 2,
            SCREEN_HEIGHT // 2 - 60,
            CONNECT_BUTTON_WIDTH,
            CONNECT_BUTTON_HEIGHT,
        )

        print(f"BombSquad Python Remote Controller")
        print(f"Targeting: {BOMBSQUAD_IP}:{BOMBSQUAD_PORT} as Player '{PLAYER_NAME}'")
        print(
            "Controls: WASD (Movement), K (Jump), J (Punch), O (Bomb), I (Pickup), Enter (Menu), LShift (Run)"
        )
        print("Close window or press ESC to quit")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if (
                        controller.connection.ui_state == UIState.DISCONNECTED
                        and connect_button_rect.collidepoint(event.pos)
                    ):
                        controller.attempt_connect()
                else:
                    controller.handle_key_event(event)

            controller.process_incoming_packets()
            controller.handle_connection_timeout()

            if controller.connection.ui_state == UIState.CONNECTED:
                controller.send_control_state_packet()

            _draw_ui(screen, controller, font_small, font_large, connect_button_rect)
            pygame.display.flip()
            clock.tick(STATE_PACKET_RATE)

    except Exception as e:
        logging.error(f"Application error: {e}")
    finally:
        controller.cleanup()
        pygame.quit()


def _draw_ui(screen, controller, font_small, font_large, connect_button_rect):
    """Draw the user interface"""
    screen.fill((30, 30, 30))

    if controller.connection.ui_state == UIState.DISCONNECTED:
        pygame.draw.rect(screen, (0, 100, 0), connect_button_rect)
        pygame.draw.rect(screen, (0, 150, 0), connect_button_rect, 2)
        text_surf = font_large.render("Connect", True, (220, 220, 220))
        text_rect = text_surf.get_rect(center=connect_button_rect.center)
        screen.blit(text_surf, text_rect)

    status_text, status_color = controller.get_status_info()
    status_surf = font_large.render(status_text, True, status_color)
    screen.blit(status_surf, (10, 10))

    info_text = f"Target: {BOMBSQUAD_IP}:{BOMBSQUAD_PORT}"
    info_surf = font_small.render(info_text, True, (150, 150, 150))
    screen.blit(info_surf, (10, 40))

    protocol_text = (
        f"Protocol: {'V2' if controller.connection.using_protocol_v2 else 'V1'}"
    )
    protocol_surf = font_small.render(protocol_text, True, (150, 150, 150))
    screen.blit(protocol_surf, (10, 60))

    if controller.connection.ui_state == UIState.CONNECTED:
        y_offset = 90
        controls_header = font_small.render("Active Controls:", True, (200, 200, 200))
        screen.blit(controls_header, (10, y_offset))
        y_offset += 20

        active_controls = controller.get_active_controls()
        if active_controls:
            for i, control in enumerate(active_controls):
                control_surf = font_small.render(control, True, (180, 180, 220))
                screen.blit(control_surf, (20, y_offset + i * 18))
        else:
            no_controls_surf = font_small.render("No actions", True, (150, 150, 150))
            screen.blit(no_controls_surf, (20, y_offset))

    esc_surf = font_small.render("Press ESC or close to quit", True, (150, 150, 150))
    screen.blit(esc_surf, (10, SCREEN_HEIGHT - 25))


if __name__ == "__main__":
    main()
