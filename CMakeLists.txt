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
set(LLAMA_CPP_DIR "" CACHE PATH "Path to a built llama.cpp source tree")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR
        "LLAMA_CPP_DIR must be set to the root of a built llama.cpp tree.\n"
        "  cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

set(LLAMA_CPP_INCLUDE      "${LLAMA_CPP_DIR}/include")
set(GGML_INCLUDE            "${LLAMA_CPP_DIR}/ggml/include")

# ───── Platform-specific library paths ──────────────────────────────────────
if(WIN32)
    # MSVC puts binaries in build/bin/Release/ and import libs in build/*/Release/
    set(LLAMA_CPP_BUILD_BIN    "${LLAMA_CPP_DIR}/build/bin/Release")
    set(LLAMA_CPP_BUILD_LIB    "${LLAMA_CPP_DIR}/build/src/Release")
    set(GGML_BUILD_LIB         "${LLAMA_CPP_DIR}/build/ggml/src/Release")

    set(LLAMA_LIB_SUFFIX       ".dll")
    set(LLAMA_IMPLIB_SUFFIX    ".lib")
else()
    set(LLAMA_CPP_BUILD_BIN    "${LLAMA_CPP_DIR}/build/bin")
    set(LLAMA_CPP_BUILD_LIB    "${LLAMA_CPP_BUILD_BIN}")
    set(GGML_BUILD_LIB         "${LLAMA_CPP_BUILD_BIN}")

    if(APPLE)
        set(LLAMA_LIB_SUFFIX   ".dylib")
    else()
        set(LLAMA_LIB_SUFFIX   ".so")
    endif()
    set(LLAMA_IMPLIB_SUFFIX    "")
endif()

if(NOT EXISTS "${LLAMA_CPP_INCLUDE}/llama.h")
    message(FATAL_ERROR "llama.h not found at ${LLAMA_CPP_INCLUDE}")
endif()

# ───── Imported targets for llama.cpp libraries ─────────────────────────────
add_library(llama SHARED IMPORTED)
if(WIN32)
    set_target_properties(llama PROPERTIES
        IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/llama${LLAMA_LIB_SUFFIX}"
        IMPORTED_IMPLIB               "${LLAMA_CPP_BUILD_LIB}/llama${LLAMA_IMPLIB_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_CPP_INCLUDE};${GGML_INCLUDE}")
else()
    set_target_properties(llama PROPERTIES
        IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/libllama${LLAMA_LIB_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_CPP_INCLUDE};${GGML_INCLUDE}")
endif()

add_library(ggml SHARED IMPORTED)
if(WIN32)
    set_target_properties(ggml PROPERTIES
        IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/ggml${LLAMA_LIB_SUFFIX}"
        IMPORTED_IMPLIB               "${GGML_BUILD_LIB}/ggml${LLAMA_IMPLIB_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE}")
else()
    set_target_properties(ggml PROPERTIES
        IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/libggml${LLAMA_LIB_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE}")
endif()

add_library(ggml-base SHARED IMPORTED)
if(WIN32)
    set_target_properties(ggml-base PROPERTIES
        IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/ggml-base${LLAMA_LIB_SUFFIX}"
        IMPORTED_IMPLIB               "${GGML_BUILD_LIB}/ggml-base${LLAMA_IMPLIB_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE}")
else()
    set_target_properties(ggml-base PROPERTIES
        IMPORTED_LOCATION             "${LLAMA_CPP_BUILD_BIN}/libggml-base${LLAMA_LIB_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE}")
endif()

# ───── Runtime library discovery (Windows: PATH, Linux: rpath) ─────────────
if(WIN32)
    # On Windows, copy llama DLLs next to our executables so they are found at runtime.
    set(LLAMA_DLLS
        "${LLAMA_CPP_BUILD_BIN}/llama.dll"
        "${LLAMA_CPP_BUILD_BIN}/ggml.dll"
        "${LLAMA_CPP_BUILD_BIN}/ggml-base.dll"
        "${LLAMA_CPP_BUILD_BIN}/ggml-cpu.dll"
    )
    # Also copy CUDA DLLs if they exist
    file(GLOB LLAMA_CUDA_DLLS "${LLAMA_CPP_BUILD_BIN}/ggml-cuda*.dll")
    set(LLAMA_ALL_DLLS ${LLAMA_DLLS} ${LLAMA_CUDA_DLLS})
else()
    set(CMAKE_INSTALL_RPATH         "${LLAMA_CPP_BUILD_BIN}")
    set(CMAKE_BUILD_RPATH           "${LLAMA_CPP_BUILD_BIN}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    if(APPLE)
        set(CMAKE_MACOSX_RPATH ON)
    endif()
endif()

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

if(WIN32)
    foreach(DLL ${LLAMA_ALL_DLLS})
        add_custom_command(TARGET moss-tts-cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" "$<TARGET_FILE_DIR:moss-tts-cli>"
            COMMENT "Copying ${DLL}")
    endforeach()
endif()

# ───── Server binary ───────────────────────────────────────────────────────
add_executable(moss-tts-server src/server/moss_tts_server.cpp)
target_link_libraries(moss-tts-server PRIVATE openmoss_core)
target_include_directories(moss-tts-server PRIVATE third_party/cpp-httplib third_party)
find_package(Threads REQUIRED)
target_link_libraries(moss-tts-server PRIVATE Threads::Threads)

if(WIN32)
    # Link ws2_32 for httplib on Windows
    target_link_libraries(moss-tts-server PRIVATE ws2_32)

    foreach(DLL ${LLAMA_ALL_DLLS})
        add_custom_command(TARGET moss-tts-server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" "$<TARGET_FILE_DIR:moss-tts-server>"
            COMMENT "Copying ${DLL}")
    endforeach()
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
