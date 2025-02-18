cmake_minimum_required(VERSION 3.5)

project(whisper_meloTTS)

# onnxruntime
set(ONNXRUNTIME_PATH ${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime)
link_directories(${ONNXRUNTIME_PATH}/lib)

# librosa
set(ROSA_PATH ${CMAKE_SOURCE_DIR}/3rd_party/librosa)

# audioFile
set(AUDIOFILE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/audioFile)

# audioFile
set(OPENCC_PATH ${CMAKE_SOURCE_DIR}/3rd_party/libopencc)

## tokenizers-cpp
#set(TOKENIZER_PATH ${CMAKE_SOURCE_DIR}/3rd_party/tokenizers-cpp)
#link_directories(${TOKENIZER_PATH}/lib)
#include_directories(${TOKENIZER_PATH}/include)

include_directories(
        ${OPENCC_PATH}/include
        ${ROSA_PATH}
        ${AUDIOFILE_PATH}
        ${ONNXRUNTIME_PATH}/include
        ${CMAKE_SOURCE_DIR}/include
)
link_directories(
        ${OPENCC_PATH}/lib
)

FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

add_library(whisper_melo STATIC ${SRCS})

target_link_libraries(whisper_melo onnxruntime dl m pthread opencc)

add_executable(demo ${CMAKE_SOURCE_DIR}/demo/demo.cpp)

target_link_libraries(demo whisper_melo)