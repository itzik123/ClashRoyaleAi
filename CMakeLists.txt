# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.10)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("ClashRoyaleEnv")

# Include sub-projects.
# add_subdirectory ("ClashRoyaleEnv")

add_executable(ClashRoyaleEnv src/main.cpp)
target_include_directories(ClashRoyaleEnv PRIVATE include/core include/entities include/rendering)

# Python Bindings with Pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(clash_royale_env src/bindings.cpp)
target_include_directories(clash_royale_env PRIVATE include/core include/entities include/rendering)

# Automatically copy and rename the generated .pyd file to the python_ai directory after compiling
add_custom_command(TARGET clash_royale_env POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:clash_royale_env> ${CMAKE_SOURCE_DIR}/python_ai/clash_royale_env.pyd
    COMMENT "Copying compiled bindings to python_ai/clash_royale_env.pyd..."
)
