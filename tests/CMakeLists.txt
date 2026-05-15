# Smoke tests — meant to validate plumbing, not correctness of generation.
# Build with: cmake --build build --target moss-tts-info

add_executable(moss-tts-info ${CMAKE_CURRENT_SOURCE_DIR}/moss_tts_info.cpp)
target_link_libraries(moss-tts-info PRIVATE openmoss_core)

add_executable(moss-tts-compute-test ${CMAKE_CURRENT_SOURCE_DIR}/moss_tts_compute.cpp)
target_link_libraries(moss-tts-compute-test PRIVATE openmoss_core)

add_executable(moss-codec-roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/moss_codec_roundtrip.cpp)
target_link_libraries(moss-codec-roundtrip PRIVATE openmoss_core)
