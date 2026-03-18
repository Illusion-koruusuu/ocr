from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    image_path_arg = DeclareLaunchArgument(
        "image_path",
        default_value="/home/lu/code/ocr/png/0.png",
        description="Path to the image for Texify OCR",
    )

    period_ms_arg = DeclareLaunchArgument(
        "period_ms",
        default_value="500",
        description="Polling period in milliseconds",
    )

    topic_arg = DeclareLaunchArgument(
        "topic",
        default_value="texify_text",
        description="Output topic for recognized LaTeX text",
    )

    hf_hub_offline_arg = DeclareLaunchArgument(
        "hf_hub_offline",
        default_value="1",
        description="Set to 1 to force Hugging Face hub offline mode",
    )

    transformers_offline_arg = DeclareLaunchArgument(
        "transformers_offline",
        default_value="1",
        description="Set to 1 to force Transformers offline mode",
    )

    return LaunchDescription(
        [
            image_path_arg,
            period_ms_arg,
            topic_arg,
            hf_hub_offline_arg,
            transformers_offline_arg,
            SetEnvironmentVariable(
                name="HF_HUB_OFFLINE",
                value=LaunchConfiguration("hf_hub_offline"),
            ),
            SetEnvironmentVariable(
                name="TRANSFORMERS_OFFLINE",
                value=LaunchConfiguration("transformers_offline"),
            ),
            Node(
                package="ocr_reader",
                executable="texify_reader_node",
                name="texify_reader_node",
                output="screen",
                parameters=[
                    {
                        "image_path": LaunchConfiguration("image_path"),
                        "period_ms": LaunchConfiguration("period_ms"),
                        "topic": LaunchConfiguration("topic"),
                    }
                ],
            ),
        ]
    )
