import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import UInt16MultiArray

from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition


SERVO_TOPIC = "servo_controller"
POT_TOPIC = "/pot_values"
SERVO_DURATION = 0.03
LOG_THROTTLE_SEC = 1.0


@dataclass(frozen=True)
class ServoMapping:
    ids: Sequence[int]

    def zip_with(self, pulse_values: Iterable[int]):
        return tuple(zip(self.ids, pulse_values))


SERVO_MAPPING = ServoMapping(ids=(10, 5, 4, 3, 2, 1))
POT_SAMPLE_COUNT = len(SERVO_MAPPING.ids)


class PotValueHandler(Node):
    def __init__(self) -> None:
        super().__init__("pot_value_handler")

        self.joints_pub = self.create_publisher(ServosPosition, SERVO_TOPIC, 1)
        self.create_subscription(UInt16MultiArray, POT_TOPIC, self.pot_callback, 10)

        self._active = threading.Event()
        self._stop = threading.Event()

        self._keyboard_thread = threading.Thread(
            target=self._keyboard_listener,
            name="keyboard-listener",
            daemon=True,
        )
        self._keyboard_thread.start()

    def destroy_node(self) -> bool:
        self._stop.set()
        if self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=0.5)
        return super().destroy_node()

    def pot_callback(self, msg: UInt16MultiArray) -> None:
        pulses = list(msg.data)
        if len(pulses) < POT_SAMPLE_COUNT:
            self.get_logger().warning(
                "Недостаточно данных потенциометра: ожидаю %d, получено %d",
                POT_SAMPLE_COUNT,
                len(pulses),
                throttle_duration_sec=LOG_THROTTLE_SEC,
            )
            return

        if self._active.is_set():
            self.get_logger().info(
                "Активный режим: передаю %s",
                pulses[:POT_SAMPLE_COUNT],
                throttle_duration_sec=LOG_THROTTLE_SEC,
            )
            servo_pairs = SERVO_MAPPING.zip_with(self._extract_servo_pulses(pulses))
            set_servo_position(self.joints_pub, SERVO_DURATION, servo_pairs)
        else:
            self.get_logger().debug(
                "Неактивный режим: манипулятор покоится",
                throttle_duration_sec=LOG_THROTTLE_SEC,
            )

    def _extract_servo_pulses(self, pulses: Sequence[int]) -> Sequence[int]:
        # Потенциометры приходят как [s1, s2, s3, s4, s5, gripper]
        # Преобразуем в ожидаемый драйвером порядок.
        ordered = [pulses[-1]] + [pulses[i] for i in range(4, -1, -1)]
        return ordered

    def _keyboard_listener(self) -> None:
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except termios.error:
            self.get_logger().error("Не удалось получить параметры терминала")
            return

        tty.setcbreak(fd)

        try:
            while not self._stop.is_set() and rclpy.ok():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                ch = sys.stdin.read(1)
                if ch == " ":
                    if self._active.is_set():
                        self._active.clear()
                        mode = "ВЫКЛЮЧЕН"
                    else:
                        self._active.set()
                        mode = "АКТИВНЫЙ"
                    print(f"\r\nРежим переключён: {mode}", flush=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = PotValueHandler()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
