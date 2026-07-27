cmake_minimum_required(VERSION 3.18)
project(openmoss CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ───── Vendored llama.cpp (provides libllama + ggml) ────────────────────────
# openmoss runs the Qwen3 backbone through libllama, so it needs the full tree,
# not just ggml. Built in-tree; the GPU backend is selected with the upstream
# flags (-DGGML_CUDA=ON / -DGGML_VULKAN=ON). Trim the parts we don't use.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)

# Colocate the shared libs (and DLLs on Windows) with our executables so they
# resolve via $ORIGIN / the exe directory, both in-tree and once packaged.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(third_party/llama.cpp)

if(NOT WIN32)
    set(CMAKE_BUILD_RPATH   "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    if(APPLE)
        set(CMAKE_MACOSX_RPATH ON)
    endif()
endif()

# ───── Common library (model + pipeline + codec) ───────────────────────────
add_library(openmoss_core STATIC
    src/model.cpp
    src/tokenizer.cpp
    src/delay.cpp
    src/frame_decoder.cpp
    src/local_transformer.cpp
    src/codec.cpp
    src/dit.cpp
    src/dac_vae.cpp
    src/soundeffect.cpp
    src/pipeline.cpp
    src/wav.cpp
)
target_include_directories(openmoss_core PUBLIC include)
target_link_libraries(openmoss_core PUBLIC llama ggml ggml-base)

# Platform-specific compile options
if(MSVC)
    target_compile_options(openmoss_core PRIVATE /W4 /utf-8)
    # Use UTF-8 for source files and execution charset
    target_compile_definitions(openmoss_core PUBLIC
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
else()
    target_compile_options(openmoss_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# ───── CLI binary ──────────────────────────────────────────────────────────
add_executable(moss-tts-cli src/cli/moss_tts_cli.cpp)
target_link_libraries(moss-tts-cli PRIVATE openmoss_core)

# ───── Server binary ───────────────────────────────────────────────────────
add_executable(moss-tts-server src/server/moss_tts_server.cpp)
target_link_libraries(moss-tts-server PRIVATE openmoss_core)
target_include_directories(moss-tts-server PRIVATE third_party/cpp-httplib third_party)
find_package(Threads REQUIRED)
target_link_libraries(moss-tts-server PRIVATE Threads::Threads)

if(WIN32)
    # Link ws2_32 for httplib on Windows
    target_link_libraries(moss-tts-server PRIVATE ws2_32)
endif()

# Make the bundled WebUI available next to the server binary so it auto-detects
# without --webui-dir. The source tree's webui/ remains the editable copy.
if(EXISTS "${CMAKE_SOURCE_DIR}/webui/index.html")
    add_custom_command(TARGET moss-tts-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/webui"
                "$<TARGET_FILE_DIR:moss-tts-server>/webui"
        COMMENT "Staging WebUI next to moss-tts-server")
endif()

# ───── Tests ───────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
