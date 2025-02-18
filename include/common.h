//
// Created by yanghaoqi on 2/17/25.
//

#pragma once

#include <vector>

constexpr int WHISPER_N_MELS = 80;
constexpr int WHISPER_SAMPLE_RATE = 16000;
constexpr int WHISPER_N_FFT = 480;
constexpr int WHISPER_HOP_LENGTH = 160;
constexpr int WHISPER_SOT = 50258;
constexpr int WHISPER_EOT = 50257;
constexpr int WHISPER_BLANK = 220;
constexpr int WHISPER_NO_TIMESTAMPS = 50363;
constexpr int WHISPER_NO_SPEECH = 50362;
constexpr int WHISPER_TRANSLATE = 50358;
constexpr int WHISPER_TRANSCRIBE = 50359;
constexpr int WHISPER_VOCAB_SIZE = 51865;
constexpr int WHISPER_N_TEXT_CTX = 448;
constexpr float NEG_INF = -INFINITY;

template <class T>
struct tensor_info
{
    std::vector<T> data;
    std::vector<long> shape;

    void update(std::vector<T> data, std::vector<long>shape)
    {
        this->data = data;
        this->shape = shape;
    }

    void update(std::vector<T> data)
    {
        this->data = data;
    }
};

template <class T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &matrix)
{
    // 获取原矩阵的行数和列数
    size_t rows = matrix.size();
    if (rows == 0)
        return {}; // 如果原矩阵为空，直接返回空矩阵
    size_t cols = matrix[0].size();

    // 创建一个新的矩阵，其行数为原矩阵的列数，列数为原矩阵的行数
    std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));

    // 对于每个元素进行转置操作
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}
