//
// Created by Curio on 2/17/25.
//

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

bool get_pos_ebd_tokens(std::string &path, std::vector<float> &positional_embedding, std::vector<std::string> &token_tables)
{
    std::string position_embd_path = "../data/tiny-positional_embedding.bin";
    positional_embedding.resize(WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE /*tiny: 384; small: 768 */);
    FILE *fp = fopen(position_embd_path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "Can NOT open %s\n", position_embd_path.c_str());
        return false;
    }
    fread(positional_embedding.data(), sizeof(float), WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE, fp);
    fclose(fp);

    std::string token_path = "../data/tiny-tokens.txt";
    std::ifstream ifs(token_path);
    if (!ifs.is_open())
    {
        fprintf(stderr, "Can NOT open %s\n", token_path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        size_t i = line.find(' ');
        token_tables.push_back(line.substr(0, i));
    }

    return true;
}

static int detect_language(const std::string &language)
{
    int i = 51; // zh
    for (int n = 0; n < WHISPER_LANG_CODES.size(); n++)
    {
        if (language == WHISPER_LANG_NAMES[n])
        {
            i = n;
            break;
        }
    }

    return WHISPER_LANG_CODES[i];
}

static void supress_tokens(std::vector<float> &logits, bool is_initial)
{
    if (is_initial)
    {
        logits[WHISPER_EOT] = NEG_INF;
        logits[WHISPER_BLANK] = NEG_INF;
    }

    logits[WHISPER_NO_TIMESTAMPS] = NEG_INF;
    logits[WHISPER_SOT] = NEG_INF;
    logits[WHISPER_NO_SPEECH] = NEG_INF;
    logits[WHISPER_TRANSLATE] = NEG_INF;
}

static int argmax(const std::vector<float> &logits)
{
    auto max_iter = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_iter); // absolute index of max
}

std::vector<std::vector<float>> load_audio(const std::string &path)
{
    AudioFile<float> audio_file;
    if (!audio_file.load(path))
    {
        printf("load wav failed!\n");
    }

    auto &samples = audio_file.samples[0];
    int n_samples = samples.size();

    auto mel = librosa::Feature::melspectrogram(samples, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, "hann",
                                                true, "reflect", 2.0f, WHISPER_N_MELS, 0.0f,
                                                WHISPER_SAMPLE_RATE / 2.0f);
    mel = transpose(mel);
    return mel;
}

std::vector<std::vector<float>> clamp_and_normlize(std::vector<std::vector<float>> &mel)
{
    int n_mel = mel.size();
    int n_len = mel[0].size();

    double mmax = -1e20;
    for (int i = 0; i < WHISPER_N_MELS; i++) {
        for (int n = 0; n < n_len; n++) {
            mel[i][n] = std::log10(std::max(mel[i][n], 1e-10f));

            if (mel[i][n] > mmax) {
                mmax = mel[i][n] ;
            }
        }
    }

    for (int i = 0; i < WHISPER_N_MELS; i++) {
        for (int n = 0; n < n_len; n++) {
            mel[i][n] = (std::max(mel[i][n], (float)(mmax - 8.0)) + 4.0)/4.0;
            mel[i].resize(3000);
        }
    }
    return mel;
}
