from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    model_dir_arg = DeclareLaunchArgument(
        "model_dir",
        default_value="/home/lu/code/ocr/onnx/tjoab_latex_finetuned",
        description="Directory containing encoder/decoder ONNX files",
    )

    image_path_arg = DeclareLaunchArgument(
        "image_path",
        default_value="/home/lu/code/ocr/png/11/roi_36.png",
        description="Input image path for ONNX smoke run",
    )

    invert_image_arg = DeclareLaunchArgument(
        "invert_image",
        default_value="true",
        description="Whether to invert image colors before inference",
    )

    max_length_arg = DeclareLaunchArgument(
        "max_length",
        default_value="128",
        description="Greedy decode max token length",
    )

    topic_arg = DeclareLaunchArgument(
        "topic",
        default_value="tjoab_token_ids",
        description="Output topic for generated token ids",
    )

    text_topic_arg = DeclareLaunchArgument(
        "text_topic",
        default_value="tjoab_text",
        description="Output topic for decoded text",
    )

    id2token_path_arg = DeclareLaunchArgument(
        "id2token_path",
        default_value="/home/lu/code/ocr/onnx/tjoab_latex_finetuned/id2token.txt",
        description="Path to id2token TSV exported from vocab.json",
    )

    period_ms_arg = DeclareLaunchArgument(
        "period_ms",
        default_value="3000",
        description="Smoke test period in milliseconds",
    )

    return LaunchDescription(
        [
            model_dir_arg,
            image_path_arg,
            invert_image_arg,
            max_length_arg,
            topic_arg,
            text_topic_arg,
            id2token_path_arg,
            period_ms_arg,
            Node(
                package="ocr_reader",
                executable="tjoab_onnx_smoke_node",
                name="tjoab_onnx_smoke_node",
                output="screen",
                parameters=[
                    {
                        "model_dir": LaunchConfiguration("model_dir"),
                        "image_path": LaunchConfiguration("image_path"),
                        "invert_image": LaunchConfiguration("invert_image"),
                        "max_length": LaunchConfiguration("max_length"),
                        "topic": LaunchConfiguration("topic"),
                        "text_topic": LaunchConfiguration("text_topic"),
                        "id2token_path": LaunchConfiguration("id2token_path"),
                        "period_ms": LaunchConfiguration("period_ms"),
                    }
                ],
            ),
        ]
    )
