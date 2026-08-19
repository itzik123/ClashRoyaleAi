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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Unit tests (Catch2), same FetchContent pattern as pybind11 above.
option(BUILD_TESTING_CLASHROYALE "Build the ClashRoyaleEnv unit test suite" ON)
if (BUILD_TESTING_CLASHROYALE)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
    )
    FetchContent_MakeAvailable(catch2)

    # Test files use the Catch2 amalgamated single-header/single-source distribution
    # (#include "catch_amalgamated.hpp"), not the modular Catch2::Catch2WithMain target,
    # so we compile extras/catch_amalgamated.cpp directly instead of linking the library.
    file(GLOB CLASHROYALE_TEST_SOURCES CONFIGURE_DEPENDS tests/entities/*.cpp tests/core/*.cpp)
    add_executable(ClashRoyaleTests ${CLASHROYALE_TEST_SOURCES} ${catch2_SOURCE_DIR}/extras/catch_amalgamated.cpp)
    target_include_directories(ClashRoyaleTests PRIVATE include/core include/entities include/rendering tests/entities ${catch2_SOURCE_DIR}/extras)

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    catch_discover_tests(ClashRoyaleTests)
endif()
