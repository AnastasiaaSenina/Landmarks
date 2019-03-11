cmake_minimum_required(VERSION 2.8)
project(Landmarks)
find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${dlib_DIRS} ${OPENCV_DIRS})

set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${dlib_LIBS})
