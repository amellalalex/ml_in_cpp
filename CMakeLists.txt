project(
    ml_in_cpp
)

add_executable(ml_in_cpp main.cc)
target_link_libraries(ml_in_cpp
    spdlog
    eigen
)
target_include_directories(ml_in_cpp PUBLIC
    spdlog/include
    eigen/Eigen
)
target_compile_options(ml_in_cpp PUBLIC
    -std=c++17
)

add_subdirectory(spdlog)
add_subdirectory(eigen)
