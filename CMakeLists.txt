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

# # gsl
# set(GSL_PATH ${CMAKE_SOURCE_DIR}/3rd_party/)

# nncase
set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase)
link_directories(${NNCASE_PATH}/lib)


## tokenizers-cpp
#set(TOKENIZER_PATH ${CMAKE_SOURCE_DIR}/3rd_party/tokenizers-cpp)
#link_directories(${TOKENIZER_PATH}/lib)
#include_directories(${TOKENIZER_PATH}/include)

include_directories(
        ${CMAKE_SOURCE_DIR}/3rd_party
        ${NNCASE_PATH}/include
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

add_library(whisper STATIC ${SRCS})

target_link_libraries(whisper PRIVATE onnxruntime dl m pthread opencc Nncase.Runtime.Native)

target_include_directories(whisper PUBLIC include)

add_executable(demo ${CMAKE_SOURCE_DIR}/demo/demo.cpp)

target_link_libraries(demo whisper)