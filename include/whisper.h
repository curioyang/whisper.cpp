//
// Created by Curio on 2/17/25.
//
#pragma once
#include "ONNXWrapper.h"
#include "base64.h"
#include "nncaseWrapper.h"
#include "wav2wav.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>

std::string whisper_get_sentence(std::string &models_dir, std::string &wav_path);