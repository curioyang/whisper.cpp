//
// Created by Curio on 2/17/25.
//

#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "wav2wav.h"
#include "ONNXWrapper.h"
#include "base64.h"

// #include "audio.h"
std::string language = "zh";

int main(int argc, const char *argv[])
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " model_dir wav_file" << std::endl;
        return 0;
    }

    std::string models_dir = argv[1];
    std::cout << "models dir is: " << models_dir << std::endl;

    std::string whisper_encoder_model = models_dir + "/tiny-encoder.onnx";
    std::string whisper_decoder_main_model = models_dir + "/tiny-decoder-main.onnx";
    std::string whisper_decoder_loop_model = models_dir + "/tiny-decoder-loop.onnx";
    std::string position_embd_path = models_dir + "/positional_embedding.bin";
    std::string tokens_path = models_dir + "/tokens.bin";

    ONNXModel whisper_encoder(std::make_unique<RuntimeManager>("whisper_encoder"), whisper_encoder_model);
    ONNXModel whisper_decoder_main(std::make_unique<RuntimeManager>("whisper_decoder_main"), whisper_decoder_main_model);
    ONNXModel whisper_decoder_loop(std::make_unique<RuntimeManager>("whisper_decoder_loop"), whisper_decoder_loop_model);

    // get audio data
    auto data  = load_audio(argv[2]);
    auto mel = clamp_and_normlize(data);


    int offset = 0;
    std::vector<float> logits(WHISPER_VOCAB_SIZE);
    int max_token_id = -1;
    std::vector<int> results;
    std::vector<long> tokens(1);
    bool is_end = false;
    std::vector<float> positional_embedding;
    std::vector<std::string> token_tables;

    get_pos_ebd_tokens(models_dir, positional_embedding, token_tables);


    // encoder
    tensor_info<float> mel_tensor;
    mel_tensor.shape = {1, (long)mel.size(), (long)mel[0].size()};
    std::vector<float> mel_data(mel_tensor.shape[1] * mel_tensor.shape[2]);
    for (int i = 0; i < mel_tensor.shape[1]; i++)
    {
        std::memcpy(mel_data.data() + i * mel_tensor.shape[2], mel[i].data(), mel_tensor.shape[2] * sizeof(float));
    }
    mel_tensor.data = mel_data;

    whisper_encoder.set_input_tensor(mel_tensor, 0);
    whisper_encoder.onForward();
    SOT_SEQUENCE[1] = detect_language(language);

    //decoder_main
    tensor_info<long> decoder_input{.data = SOT_SEQUENCE, .shape = {1, 4}};
    auto decode_main_k = whisper_encoder.get_result_vector<float>(0);
    auto decode_main_v = whisper_encoder.get_result_vector<float>(1);
    whisper_decoder_main.set_input_tensor(decoder_input, 0);
    whisper_decoder_main.set_input_tensor(decode_main_k, 1);
    whisper_decoder_main.set_input_tensor(decode_main_v, 2);
    whisper_decoder_main.onForward();

    auto decoder_main_logits = whisper_decoder_main.get_result_vector<float>(0);
    offset += decoder_input.data.size();

    std::copy(decoder_main_logits.data.begin() + 3 * WHISPER_VOCAB_SIZE, decoder_main_logits.data.end(), logits.begin());
    supress_tokens(logits, true);
    max_token_id = argmax(logits);

    std::vector<float> mask(WHISPER_N_TEXT_CTX);
    for (int n = 0; n < WHISPER_N_TEXT_CTX - offset - 1; n++)
    {
        mask[n] = NEG_INF;
    }

    std::vector<float> part_pos_data(positional_embedding.begin() + offset * WHISPER_N_TEXT_STATE, positional_embedding.begin() + (offset+1) * WHISPER_N_TEXT_STATE);

    tensor_info<long> tokens_tensor{.data = tokens, .shape = {1, 1}};

    tensor_info<float> in_n_layer_self_k_cache = whisper_decoder_main.get_result_vector<float>(1);
    tensor_info<float> in_n_layer_self_v_cache = whisper_decoder_main.get_result_vector<float>(2);
    tensor_info<float> n_layer_cross_k = whisper_encoder.get_result_vector<float>(0);
    tensor_info<float> n_layer_cross_v = whisper_encoder.get_result_vector<float>(1);

    tensor_info<float> positional_embedding_tensor{.data = part_pos_data, .shape = {1, (long)part_pos_data.size()}};
    tensor_info<float> mask_tensor{.data = mask, .shape = {WHISPER_N_TEXT_CTX}};

    for (int i = 0; i < WHISPER_N_TEXT_CTX - 4; i++)
    {
        if(max_token_id == WHISPER_EOT)
        {
            is_end = true;
            break;
        }
        results.emplace_back(max_token_id);
        tokens[0] = results.back();

        tokens_tensor.update(tokens);

        whisper_decoder_loop.set_input_tensor(tokens_tensor, 0);
        whisper_decoder_loop.set_input_tensor(in_n_layer_self_k_cache, 1);
        whisper_decoder_loop.set_input_tensor(in_n_layer_self_v_cache, 2);
        whisper_decoder_loop.set_input_tensor(n_layer_cross_k, 3);
        whisper_decoder_loop.set_input_tensor(n_layer_cross_v, 4);
        whisper_decoder_loop.set_input_tensor(positional_embedding_tensor, 5);
        whisper_decoder_loop.set_input_tensor(mask_tensor, 6);

        whisper_decoder_loop.onForward();

        in_n_layer_self_k_cache = whisper_decoder_loop.get_result_vector<float>(1);
        in_n_layer_self_v_cache = whisper_decoder_loop.get_result_vector<float>(2);

        logits = whisper_decoder_loop.get_result_vector<float>(0).data;

        offset += 1;
        mask_tensor.data[WHISPER_N_TEXT_CTX - offset - 1] = 0;

        supress_tokens(logits, false);
        max_token_id = argmax(logits);
        // n_layer_cross_k.update()
        // n_layer_cross_v.update()
        // positional_embedding.update()
        
    }

    std::string s;
    for (const auto i : results)
    {
        char str[1024];
        base64_decode((const uint8 *)token_tables[i].c_str(), (uint32)token_tables[i].size(), str);
        s += str;
    }

    if (language == "en")
        printf("Result: %s\n", s.c_str());
    else
    {
        const opencc::SimpleConverter converter("t2s.json");
        std::string simple_str = converter.Convert(s);
        printf("Result: %s\n", simple_str.c_str());
    }

    return 0;
}
