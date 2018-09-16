cmake_minimum_required(VERSION 3.1)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/nlopt-util.hpp DESTINATION include)

find_package(Eigen3)

add_library(nlopt-util INTERFACE)
target_sources(nlopt-util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/nlopt-util.hpp)
target_include_directories(nlopt-util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nlopt-util INTERFACE Eigen3::Eigen)
