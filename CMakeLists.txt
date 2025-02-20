cmake_minimum_required(VERSION 3.5)

project(whisper_meloTTS)

# onnxruntime
set(ONNXRUNTIME_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/onnxruntime)
link_directories(${ONNXRUNTIME_PATH}/lib)

# librosa
set(ROSA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/librosa)

# audioFile
set(AUDIOFILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/audioFile)

# openCC
set(OPENCC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/libopencc)
find_library(OPENCC_LIB NAMES opencc PATHS ${OPENCC_PATH}/lib)
link_directories(${OPENCC_PATH}/lib)


# # gsl
# set(GSL_PATH ${CMAKE_SOURCE_DIR}/3rd_party/)

# nncase
set(NNCASE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/nncase)
find_library(NNCASE_LIB NAMES Nncase.Runtime.Native PATHS ${NNCASE_PATH}/lib)
link_directories(${NNCASE_PATH}/lib)

## tokenizers-cpp
#set(TOKENIZER_PATH ${CMAKE_SOURCE_DIR}/3rd_party/tokenizers-cpp)
#link_directories(${TOKENIZER_PATH}/lib)
#include_directories(${TOKENIZER_PATH}/include)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party
        ${NNCASE_PATH}/include
        ${OPENCC_PATH}/include
        ${ROSA_PATH}
        ${AUDIOFILE_PATH}
        ${ONNXRUNTIME_PATH}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

add_library(whisper STATIC ${SRCS})

target_link_directories(whisper PRIVATE ${OPENCC_PATH}/lib ${NNCASE_PATH}/lib)

target_link_libraries(whisper PRIVATE onnxruntime dl m pthread  ${OPENCC_LIB} ${NNCASE_LIB})

install(TARGETS whisper
    EXPORT whisper_export
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

export(EXPORT whisper_export
    FILE ${CMAKE_CURRENT_BINARY_DIR}/whisper_targets.cmake
)

add_executable(demo ${CMAKE_CURRENT_SOURCE_DIR}/demo/demo.cpp)

target_link_libraries(demo whisper)