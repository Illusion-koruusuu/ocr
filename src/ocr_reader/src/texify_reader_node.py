#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PIL import Image
import os
import torch
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor

class TexifyReaderNode(Node):
    def __init__(self):
        super().__init__('texify_reader_node')
        self.get_logger().info('Initializing Texify Reader Node...')

        # Declare parameters
        self.declare_parameter('image_path', '/png/test.png')
        self.declare_parameter('period_ms', 500)
        self.declare_parameter('topic', 'texify_text')

        # Get parameters
        self.image_path = self.get_parameter('image_path').get_parameter_value().string_value
        period_ms = self.get_parameter('period_ms').get_parameter_value().integer_value
        topic = self.get_parameter('topic').get_parameter_value().string_value

        # Initialize Texify model
        self.get_logger().info('Loading Texify model...')
        # Check if CUDA is available and use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        self.model = load_model(device=self.device)
        self.processor = load_processor()
        self.get_logger().info('Texify model loaded successfully.')

        # Create publisher and timer
        self.publisher_ = self.create_publisher(String, topic, 10)
        self.timer = self.create_timer(period_ms / 1000.0, self.tick)
        self.get_logger().info(f'Node initialized. Watching {self.image_path} and publishing to {topic}.')

    def tick(self):
        if not os.path.exists(self.image_path):
            self.get_logger().warn(f'Image not found at: {self.image_path}')
            return

        try:
            img = Image.open(self.image_path)
            latex_result = batch_inference([img], self.model, self.processor)

            # The result is a list, get the first element
            text = latex_result[0] if latex_result else ""

            msg = String()
            msg.data = text
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published LaTeX: {text}')

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

def main(argc=None):
    rclpy.init(args=argc)
    texify_reader_node = TexifyReaderNode()
    try:
        rclpy.spin(texify_reader_node)
    except KeyboardInterrupt:
        pass
    finally:
        texify_reader_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
