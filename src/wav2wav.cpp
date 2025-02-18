//
// Created by Curio on 2/17/25.
//

#include "wav2wav.h"



// auto clamp_and_normlize()
// {
//     // clamping and normalization
//     double mmax = -1e20;
//     for (int i = 0; i < WHISPER_N_MELS; i++) {
//         for (int n = 0; n < n_len; n++) {
//             mel[i][n] = std::log10(std::max(mel[i][n], 1e-10f));

//             if (mel[i][n] > mmax) {
//                 mmax = mel[i][n] ;
//             }
//         }
//     }

//     for (int i = 0; i < WHISPER_N_MELS; i++) {
//         for (int n = 0; n < n_len; n++) {
//             mel[i][n] = (std::max(mel[i][n], (float)(mmax - 8.0)) + 4.0)/4.0;
//             mel[i].resize(3000);
//         }
//     }

//     n_len = mel[0].size();
// }