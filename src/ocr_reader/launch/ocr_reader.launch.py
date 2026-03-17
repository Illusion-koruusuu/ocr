from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    image_path_arg = DeclareLaunchArgument(
        "image_path",
        default_value="/home/lu/code/ocr/input.png",
        description="Absolute path to the image to OCR",
    )

    return LaunchDescription(
        [
            image_path_arg,
            Node(
                package="ocr_reader",
                executable="ocr_reader_node",
                name="ocr_reader",
                output="screen",
                parameters=[{"image_path": LaunchConfiguration("image_path")}],
            ),
        ]
    )
