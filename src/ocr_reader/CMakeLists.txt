cmake_minimum_required(VERSION 3.8)
project(ocr_reader)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPT REQUIRED lept)

add_executable(ocr_reader_node src/ocr_reader_node.cpp)
target_include_directories(ocr_reader_node PRIVATE ${TESSERACT_INCLUDE_DIRS} ${LEPT_INCLUDE_DIRS})
target_link_libraries(ocr_reader_node ${TESSERACT_LIBRARIES} ${LEPT_LIBRARIES})
ament_target_dependencies(ocr_reader_node rclcpp std_msgs)

install(TARGETS ocr_reader_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(
  PROGRAMS src/texify_reader_node.py
  RENAME texify_reader_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
