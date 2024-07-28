cmake_minimum_required(VERSION 3.25)

project(DelaunayTriangulation)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -Wall")

add_executable(Main main.cpp)

include(F:/vcpkg/scripts/buildsystems/vcpkg.cmake)

find_package(Mathter REQUIRED)
include_directories(${Mathter_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)

set(CMAKE_PREFIX_PATH "F:/glew-2.1.0")
set(CMAKE_LIBRARY_PATH "F:/glew-2.1.0/lib/Release/Win32/")
find_package(GLEW REQUIRED)

set(GLFWPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("F:/glfwpp" "glfwpp_build")

find_package(Mathter REQUIRED)

target_link_libraries(Main PRIVATE GLFWPP)
target_link_libraries(Main PRIVATE OpenGL::GL)
target_link_libraries(Main PRIVATE GLEW::GLEW)
target_link_libraries(Main PRIVATE Mathter::Mathter)
target_link_libraries(Main ${Mathter_LIBRARIES})