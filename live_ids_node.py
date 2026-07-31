import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import torch
import numpy as np
from collections import deque
from model import VehicularGRU_IDS

class RealTimeGazeboIDS(Node):
    def __init__(self):
        super().__init__('realtime_gazebo_ids')
       
        # Load the compiled 9-feature model file
        self.model = VehicularGRU_IDS(input_dim=9)
        self.model.load_state_dict(torch.load("global_road_ids_model.pth"))
        self.model.eval()
       
        self.sequence_length = 10
        self.telemetry_history = deque(maxlen=self.sequence_length)
       
        # Intercept the primary Gazebo movement engine topic
        self.subscription = self.create_subscription(
            Twist,
            '/model/vehicle/cmd_vel',
            self.listener_callback,
            10)
        self.get_logger().info("🛡️ CICIoV2024 GRU Engine Active. Intercepting Gazebo Streams...")

    def listener_callback(self, msg):
        # Convert ROS2 geometry fields into an array structured like the decimal dataset rows
        simulated_can_id = 0x254
        b0 = min(max(int(abs(msg.linear.x) * 20), 0), 255)
        b1 = min(max(int(abs(msg.angular.z) * 20), 0), 255)
       
        current_frame = [simulated_can_id, b0, b1, 0, 0, 0, 0, 0, 0]
        self.telemetry_history.append(current_frame)
       
        if len(self.telemetry_history) == self.sequence_length:
            input_tensor = torch.tensor(np.array(self.telemetry_history, dtype=np.float32)).unsqueeze(0)
           
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
               
            if prediction == 1:
                self.get_logger().error(f"🚨 INTRUSION BLOCK: Anomaly pattern intercepted! Malicious Certainty: {probabilities[0][1].item()*100:.2f}%")
            else:
                self.get_logger().info("Telemetry nominal. Packets safe.")

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