cmake_minimum_required(VERSION 3.18)
project(openmoss CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ───── Locate llama.cpp ─────────────────────────────────────────────────────
# Expect a built tree with libllama.so, libggml*.so, and headers.
set(LLAMA_CPP_DIR "" CACHE PATH "Path to a built llama.cpp source tree")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR
        "LLAMA_CPP_DIR must be set to the root of a built llama.cpp tree.\n"
        "  cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

set(LLAMA_CPP_INCLUDE      "${LLAMA_CPP_DIR}/include")
set(GGML_INCLUDE            "${LLAMA_CPP_DIR}/ggml/include")
set(LLAMA_CPP_BUILD_BIN    "${LLAMA_CPP_DIR}/build/bin")

if(NOT EXISTS "${LLAMA_CPP_INCLUDE}/llama.h")
    message(FATAL_ERROR "llama.h not found at ${LLAMA_CPP_INCLUDE}")
endif()
if(NOT EXISTS "${LLAMA_CPP_BUILD_BIN}/libllama.so")
    message(FATAL_ERROR
        "libllama.so not found at ${LLAMA_CPP_BUILD_BIN}.\n"
        "Build llama.cpp first: cd ${LLAMA_CPP_DIR} && cmake -B build && cmake --build build -j")
endif()

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/libllama.so"
    INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_CPP_INCLUDE};${GGML_INCLUDE}")

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES
    IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/libggml.so"
    INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE}")

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES
    IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/libggml-base.so"
    INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE}")

# At runtime, the executable needs to find the llama.cpp shared libs alongside our binary
# (or via LD_LIBRARY_PATH). Using rpath simplifies development setups.
set(CMAKE_INSTALL_RPATH         "${LLAMA_CPP_BUILD_BIN}")
set(CMAKE_BUILD_RPATH           "${LLAMA_CPP_BUILD_BIN}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ───── Common library (model + pipeline + codec) ───────────────────────────
add_library(openmoss_core STATIC
    src/model.cpp
    src/tokenizer.cpp
    src/delay.cpp
    src/codec.cpp
    src/pipeline.cpp
    src/wav.cpp
)
target_include_directories(openmoss_core PUBLIC include)
target_link_libraries(openmoss_core PUBLIC llama ggml ggml-base)
target_compile_options(openmoss_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

# ───── CLI binary ──────────────────────────────────────────────────────────
add_executable(moss-tts-cli src/cli/moss_tts_cli.cpp)
target_link_libraries(moss-tts-cli PRIVATE openmoss_core)

# ───── Server binary ───────────────────────────────────────────────────────
add_executable(moss-tts-server src/server/moss_tts_server.cpp)
target_link_libraries(moss-tts-server PRIVATE openmoss_core)
target_include_directories(moss-tts-server PRIVATE third_party/cpp-httplib)
find_package(Threads REQUIRED)
target_link_libraries(moss-tts-server PRIVATE Threads::Threads)

# ───── Tests ───────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
