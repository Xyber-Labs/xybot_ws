import rclpy
from rclpy.node import Node
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from std_msgs.msg import Empty, String
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import queue
import time
import uvicorn
import playsound3
from playsound3 import playsound
import os
import numpy as np

class Command(BaseModel):
    commandType: str
    parameters: Dict[str, Any]


class CommandRequest(BaseModel):
    sequence: List[Command]
    correlationId: str


class CommandResponse(BaseModel):
    status: str
    completion_time_seconds: Optional[float] = None
    error: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str
    active_task_id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None

action_completion_time = {
    'PoC1': 9.1,
    'PoC2': 10.1,
    'PoC3': 11.4,
    'PoC4': 5.8,
    'PoC5': 10.1,
    'PoC6': 11.4,
    'PoC7': 6.8,
    'PoC8': 10.4,
    'PoC9': 24.4,
    'PoC_idle_1' : 11.2,
    'PoC_idle_2' : 30,
    }



class ApiServerNode(Node):
    def __init__(self):
        super().__init__('api_server_node')
        self.get_logger().info('API Server Node has been started.')
        self.app = FastAPI()
        self.last_sound = None
        self.error_cb_group = MutuallyExclusiveCallbackGroup()
        self.queue_empty = True
        self.last_action_time = 0
        self.action_wait_time = 0
        # Queue for sequential command execution
        self.command_queue: queue.Queue = queue.Queue()
        self.current_task_id: Optional[str] = None
        self.status: str = 'None'
        self.controller_status: str = 'None'
        self.error: str = None
        self.time_of_last_command = time.time()
        self.estimated_time = 0
        self.is_got_chest_command = False
        # Endpoints following PDF spec (Robot Commands & Status) citeturn1file0
        self.app.post(
            "/api/robot/command", response_model=CommandResponse
        )(self.handle_command)
        self.app.get(
            "/api/robot/status", response_model=StatusResponse
        )(self.handle_status)
        self.create_publishers()
        self.create_subscribers()


        # Background thread to process queued commands sequentially
        self.processing_thread = Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()


    def create_publishers(self):
        self.chest_pub = self.create_publisher(Empty, '/poc/chest', 1)
        self.reset_pub = self.create_publisher(Empty, '/poc/reset', 1)
        self.action_pub = self.create_publisher(String, '/poc/action', 1)
        self.stop_idle_pub = self.create_publisher(Empty, '/poc/stop_idle', 1)

    def create_subscribers(self):
        self.create_subscription(String, '/poc/error', self.error_callback, 1, callback_group=self.error_cb_group)
        self.create_subscription(String, '/poc/status', self.status_callback, 1, callback_group = ReentrantCallbackGroup())

    def status_callback(self, msg: String):
        self.controller_status = msg.data


    def error_callback(self, msg: String):
        # Handle error messages from the robot
        self.get_logger().error(f"Error from robot: {msg.data}")
        self.error = msg.data
        self.status = 'ERROR'
        self.current_task_id = None


    def handle_command(self, req: CommandRequest) -> CommandResponse:
        # Enqueue each action in the sequence for sequential execution
        self.estimated_time = 0
        self.stop_idle_pub.publish(Empty())
        time.sleep(0.5)
        for cmd in req.sequence:
            self.command_queue.put((req.correlationId, cmd))
            if cmd.commandType == 'ACTION':
                if self.estimated_time == 0:
                    self.estimated_time = action_completion_time.get(cmd.parameters['name'], 0)
                else:
                    self.estimated_time = self.estimated_time if self.estimated_time > action_completion_time.get(cmd.parameters['name'], 0) else action_completion_time.get(cmd.parameters['name'], 0)
            elif cmd.commandType == 'WAIT':
                self.estimated_time += cmd.parameters['seconds']
            elif cmd.commandType == 'RESET':
                self.estimated_time += 5
            elif cmd.commandType == 'OPEN_CHEST':
                self.estimated_time += 5
            elif cmd.commandType == 'PLAY_AUDIO':
                if self.estimated_time == 0:
                    self.estimated_time += cmd.parameters.get('duration', 0)
                else:
                    self.estimated_time = self.estimated_time if self.estimated_time > cmd.parameters.get('duration', 0) else cmd.parameters.get('duration', 0)
            else:
                self.get_logger().error(f'Unknown command type: {cmd.commandType}')
                self.error = {'message': 'Unknown command type'}
                return CommandResponse(status='error', error=self.error)
        time.sleep(1)  # Allow time for the command to be processed
        return CommandResponse(status='ok', completion_time_seconds=self.estimated_time, error=self.error)

    def handle_status(self) -> StatusResponse:
        # Return current execution status and active task
        return StatusResponse(
            status=self.status,
            active_task_id=self.current_task_id,
            error=self.error
        )

    def process_commands(self):
        # Continuously process commands in FIFO order
        while rclpy.ok():
            if self.controller_status != 'error':
                try:
                    corr_id, cmd = self.command_queue.get(timeout=1.0)
                    self.current_task_id = corr_id
                    self.status = f'EXECUTING {corr_id}'
                    self.execute_command(cmd)
                    self.command_queue.task_done()
                    self.get_logger().info(f'Command {cmd.commandType} executed successfully')
                    self.time_of_last_command = time.time()
                except queue.Empty:
                    # No pending commands: set status to IDLE
                    if self.status != 'IDLE' and not self.is_got_chest_command and time.time() - self.time_of_last_command > self.estimated_time + 5:
                        self.get_logger().info(f'No commands in queue for 5 seconds, setting status to IDLE')
                        self.status = 'IDLE'
                        self.current_task_id = None
                        self.idle_action = np.random.choice(['PoC_idle_1', 'PoC_idle_2'])
                    if self.status == 'IDLE' and time.time() - self.last_action_time > self.action_wait_time + 0.5:
                        self.handle_action({"name" : self.idle_action})
                    time.sleep(0.1)

    def execute_command(self, cmd: Command):
        # Dispatch based on commandType; stubs to be implemented
        cmd_type = cmd.commandType
        params = cmd.parameters
        command_dict = {
            'PLAY_AUDIO': self.handle_play_audio,
            'WAIT': self.handle_wait,
            'OPEN_CHEST': self.handle_open_chest,
            'RESET': self.handle_reset,
            'ACTION': self.handle_action,
        }
        if cmd_type in command_dict:
            command_dict[cmd_type](params)
        else:
            self.get_logger().error(f'Unknown command type: {cmd_type}')

    def handle_play_audio(self, params: Dict[str, Any]):
        self.last_sound.stop() if self.last_sound else None
        self.get_logger().info(f'Playing audio with params: {params}')
        volume = params.get('volume', 100)
        os.system(f'pactl set-sink-volume 1 {volume}%')
        os.system(f'pactl set-sink-volume 0 {volume}%')
        self.last_sound = playsound(params['audioFileUrl'], block=False)

    def handle_wait(self, params: Dict[str, Any]):
        sec = params['seconds']
        self.get_logger().info(f'Waiting {sec} second(-s)')
        time.sleep(sec)

    def handle_open_chest(self, params: Dict[str, Any]):
        # time.sleep(self.estimated_time)
        self.get_logger().info(f'Opening chest')
        self.is_got_chest_command = True
        # УБРАТЬ КОГДА КАМЕРА ВЕРНЕТСЯ
        self.chest_pub.publish(Empty())

    def handle_reset(self, params: Dict[str, Any]):
        self.get_logger().info(f'Resetting robot')
        self.reset_pub.publish(Empty())

    def handle_action(self, params: Dict[str, Any]):
        action = params["name"]
        self.get_logger().info(f'Executing action: {action}')
        self.action_pub.publish(String(data=action))
        self.action_wait_time = action_completion_time.get(action, 0)
        self.last_action_time = time.time()
        self.get_logger().info(f'Action {action} completed')


def start_api_server(app: FastAPI):
    uvicorn.run(app, host="0.0.0.0", port=7890)


def main():
    rclpy.init()
    node = ApiServerNode()

    # Run FastAPI server alongside ROS2 node
    api_thread = Thread(target=start_api_server, daemon=True, args=(node.app,))
    api_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
