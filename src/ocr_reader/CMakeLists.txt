cmake_minimum_required(VERSION 3.8)
project(ocr_reader)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPT REQUIRED lept)

set(ONNXRUNTIME_ROOT "/home/lu/onnxruntime-linux-x64-1.24.4" CACHE PATH "Path to ONNX Runtime root directory")
if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime root directory" FORCE)
endif()

if(ONNXRUNTIME_ROOT)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
  )

  find_library(ONNXRUNTIME_LIB
    onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
  )
else()
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
  find_library(ONNXRUNTIME_LIB onnxruntime)
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB AND NOT TARGET onnxruntime::onnxruntime)
  add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
  message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
  message(WARNING "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to use C++ ONNX inference later.")
endif()

add_executable(ocr_reader_node src/ocr_reader_node.cpp)
target_include_directories(ocr_reader_node PRIVATE ${TESSERACT_INCLUDE_DIRS} ${LEPT_INCLUDE_DIRS})
target_link_libraries(ocr_reader_node ${TESSERACT_LIBRARIES} ${LEPT_LIBRARIES})
ament_target_dependencies(ocr_reader_node rclcpp std_msgs)

if(TARGET onnxruntime::onnxruntime)
  add_executable(tjoab_onnx_smoke_node src/tjoab_onnx_smoke_node.cpp)
  target_include_directories(tjoab_onnx_smoke_node PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(tjoab_onnx_smoke_node onnxruntime::onnxruntime)
  target_link_libraries(tjoab_onnx_smoke_node ${OpenCV_LIBS})
  ament_target_dependencies(tjoab_onnx_smoke_node rclcpp std_msgs)
endif()

install(TARGETS ocr_reader_node DESTINATION lib/${PROJECT_NAME})

if(TARGET tjoab_onnx_smoke_node)
  install(TARGETS tjoab_onnx_smoke_node DESTINATION lib/${PROJECT_NAME})
endif()

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(
  PROGRAMS src/texify_reader_node.py
  RENAME texify_reader_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
