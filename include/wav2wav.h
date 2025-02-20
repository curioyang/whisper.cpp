//
// Created by Curio on 2/17/25.
//
#pragma once
#include "librosa.h"
#include "common.h"
#include "audioFile.h"
#include "opencc/opencc.h"

constexpr int WHISPER_N_TEXT_STATE = 384;

static std::vector<long> SOT_SEQUENCE{WHISPER_SOT,50260,WHISPER_TRANSCRIBE,WHISPER_NO_TIMESTAMPS};

static std::vector<int> WHISPER_LANG_CODES{
    50273,50303,50288,50261,50342,50299,50330,50302,50336,50267,50287,50292,50294,50323,50348,50291,50317,
    50326,50289,50356,50290,50282,50347,50331,50354,50264,50333,50296,50339,50318,50305,50293,50280,50322,
    50312,50306,50353,50285,50275,50340,50278,50268,50337,50316,50266,50307,50310,50338,50334,50313,50351,
    50260,50344,50283,50327,50272,50324,50276,50281,50301,50332,50300,50309,50343,50349,50335,50320,50259,
    50284,50304,50277,50311,50319,50314,50352,50328,50286,50274,50329,50270,50269,50350,50263,50345,50298,
    50279,50297,50262,50315,50321,50308,50355,50265,50346,50295,50271,50357,50341,50325
};

static std::vector<std::string> WHISPER_LANG_NAMES{
    "sv","sr","no","de","nn","te", "be","bn","lo","pt","ta","bg","la","km","tl","hr","sq","so","th","jw","ur","ms","bo",
    "tg","ha","ko","gu","ml","ht", "sw","sl","lt","uk","si","hy","kn","ln","da","id","ps","vi","tr","uz","kk","ja","et",
    "eu","fo","am","ne","tt","zh", "sa","cs","af","ar","sn","hi","el","lv","sd","fa","br","mt","mg","yi","mr","en","ro",
    "az","fi","is","gl","mn","haw","oc","hu","it","ka","ca","pl","as","ru","lb","sk","he","cy","es","bs","pa","mk","ba",
    "fr","my","mi","nl","su","tk", "yo"
};

bool get_pos_ebd_tokens(std::string &path, std::vector<float> &positional_embedding, std::vector<std::string> &token_tables);

int detect_language(const std::string &language);

void supress_tokens(std::vector<float> &logits, bool is_initial);

int argmax(const std::vector<float> &logits);

std::vector<std::vector<float>> load_audio(const std::string &path);

std::vector<std::vector<float>> clamp_and_normlize(std::vector<std::vector<float>> &mel);
