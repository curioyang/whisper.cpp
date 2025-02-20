
#include "wav2wav.h"

bool get_pos_ebd_tokens(std::string &path, std::vector<float> &positional_embedding, std::vector<std::string> &token_tables)
{
    std::string position_embd_path = path + "/data/tiny-positional_embedding.bin";
    positional_embedding.resize(WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE /*tiny: 384; small: 768 */);
    FILE *fp = fopen(position_embd_path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "Can NOT open %s\n", position_embd_path.c_str());
        return false;
    }
    fread(positional_embedding.data(), sizeof(float), WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE, fp);
    fclose(fp);

    std::string token_path = path + "/data/tiny-tokens.txt";
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

int detect_language(const std::string &language)
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

void supress_tokens(std::vector<float> &logits, bool is_initial)
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

int argmax(const std::vector<float> &logits)
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
    for (int i = 0; i < WHISPER_N_MELS; i++)
    {
        for (int n = 0; n < n_len; n++)
        {
            mel[i][n] = std::log10(std::max(mel[i][n], 1e-10f));

            if (mel[i][n] > mmax)
            {
                mmax = mel[i][n];
            }
        }
    }

    for (int i = 0; i < WHISPER_N_MELS; i++)
    {
        for (int n = 0; n < n_len; n++)
        {
            mel[i][n] = (std::max(mel[i][n], (float)(mmax - 8.0)) + 4.0) / 4.0;
            mel[i].resize(3000);
        }
    }
    return mel;
}