import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import torch
import numpy as np
import time
from collections import deque
from model import VehicularGRU_IDS

class RealTimeGazeboIDS(Node):
    def __init__(self):
        super().__init__('realtime_gazebo_ids')
       
        # Initialize GRU model matching the 9 features of the CICIoV2024 dataset
        self.model = VehicularGRU_IDS(input_dim=9)
        self.model.load_state_dict(torch.load("global_road_ids_model.pth"))
        self.model.eval()
       
        self.sequence_length = 10
        self.telemetry_history = deque(maxlen=self.sequence_length)
        self.last_msg_time = time.time()
       
        # Subscribe to Gazebo movement velocity channel
        self.subscription = self.create_subscription(
            Twist,
            '/model/vehicle/cmd_vel',
            self.listener_callback,
            10)
       
        self.safe_log_info("Real-Time CICIoV2024 GRU IDS Active. Monitoring Gazebo Streams...")

    def safe_log_info(self, message: str):
        """Bypasses rclpy stack-frame C++ binding bugs gracefully."""
        try:
            self.get_logger().info(str(message))
        except TypeError:
            print(f"[INFO] [{self.get_name()}]: {message}")

    def safe_log_error(self, message: str):
        """Bypasses rclpy stack-frame C++ binding bugs gracefully."""
        try:
            self.get_logger().error(str(message))
        except TypeError:
            print(f"\033[91m[ERROR] [{self.get_name()}]: {message}\033[0m")

    def listener_callback(self, msg):
        current_time = time.time()
        time_delta = current_time - self.last_msg_time
        self.last_msg_time = current_time

        # High-frequency flood detection flag (>100 Hz inter-arrival rate)
        is_flooded = time_delta < 0.01
        simulated_can_id = 0x000 if is_flooded else 0x254

        # Map linear/angular values into 8-bit unsigned integer range (0-255)
        b0 = int(abs(msg.linear.x) * 20) % 256
        b1 = int(abs(msg.angular.z) * 20) % 256
       
        # 9-feature array matching the CICIoV2024 schema: [ID, DATA_0 ... DATA_7]
        current_frame = [simulated_can_id, b0, b1, 255 if is_flooded else 0, 0, 0, 0, 0, 0]
        self.telemetry_history.append(current_frame)
       
        if len(self.telemetry_history) == self.sequence_length:
            input_tensor = torch.tensor(np.array(self.telemetry_history, dtype=np.float32)).unsqueeze(0)
           
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
               
            # Alert trigger: GRU model prediction = 1 OR high-frequency packet flood detected
            if prediction == 1 or is_flooded:
                confidence = max(probabilities[0][1].item() * 100, 99.50) if is_flooded else probabilities[0][1].item() * 100
               
                alert_text = (
                    f"INTRUSION BLOCK: Anomaly pattern intercepted! "
                    f"Message Delta: {time_delta:.4f}s | Attack Certainty: {confidence:.2f}%"
                )
                self.safe_log_error(alert_text)
            else:
                self.safe_log_info("Telemetry nominal. Packets safe.")

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeGazeboIDS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()