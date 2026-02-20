#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ap_int.h>

// -------------------------------------------------
// Types (must match kernel)
// -------------------------------------------------
typedef ap_int<8>  int8;
typedef ap_int<32> int32;

// -------------------------------------------------
// Kernel declaration
// -------------------------------------------------
extern "C" {
void yolo_conv_kv260(
    int8  *in_fm,
    int8  *out_fm,
    int8  *weights,
    int32 *bias
);
}

// -------------------------------------------------
// SAME CONFIG AS KERNEL
// -------------------------------------------------
#define H       26
#define W       26
#define CIN     128
#define COUT    128
#define K       3

#define OH (H - K + 1)
#define OW (W - K + 1)

// -------------------------------------------------
// Golden reference convolution (CPU)
// -------------------------------------------------
void golden_conv(
    int8  *in_fm,
    int8  *out_ref,
    int8  *weights,
    int32 *bias
) {
    for (int co = 0; co < COUT; co++) {
        for (int y = 0; y < OH; y++) {
            for (int x = 0; x < OW; x++) {

                int32 acc = bias[co];

                for (int ci = 0; ci < CIN; ci++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {

                            int in_idx =
                                ((y + ky) * W + (x + kx)) * CIN + ci;

                            int w_idx =
                                co * CIN * K * K +
                                ci * K * K +
                                ky * K + kx;

                            acc += in_fm[in_idx] * weights[w_idx];
                        }
                    }
                }

                acc >>= 8;
                if (acc > 127)  acc = 127;
                if (acc < -128) acc = -128;

                int out_idx =
                    (y * OW + x) * COUT + co;

                out_ref[out_idx] = (int8)acc;
            }
        }
    }
}

// -------------------------------------------------
// MAIN TEST
// -------------------------------------------------
int main() {

    std::cout << "Starting YOLO Conv KV260 Testbench...\n";

    // -------------------------------------------------
    // Allocate memory
    // -------------------------------------------------
    int8  *in_fm   = new int8 [H * W * CIN];
    int8  *out_hw  = new int8 [OH * OW * COUT];
    int8  *out_ref = new int8 [OH * OW * COUT];
    int8  *weights = new int8 [COUT * CIN * K * K];
    int32 *bias    = new int32[COUT];

    // -------------------------------------------------
    // Initialize inputs (deterministic)
    // -------------------------------------------------
    for (int i = 0; i < H * W * CIN; i++)
        in_fm[i] = (i % 13) - 6;

    for (int i = 0; i < COUT * CIN * K * K; i++)
        weights[i] = (i % 7) - 3;

    for (int i = 0; i < COUT; i++)
        bias[i] = i % 16;

    memset(out_hw,  0, OH * OW * COUT);
    memset(out_ref, 0, OH * OW * COUT);

    // -------------------------------------------------
    // Run kernel
    // -------------------------------------------------
    std::cout << "Running HLS kernel...\n";
    yolo_conv_kv260(in_fm, out_hw, weights, bias);

    // -------------------------------------------------
    // Run golden reference
    // -------------------------------------------------
    std::cout << "Running golden reference...\n";
    golden_conv(in_fm, out_ref, weights, bias);

    // -------------------------------------------------
    // Compare results
    // -------------------------------------------------
    int errors = 0;
    for (int i = 0; i < OH * OW * COUT; i++) {
        if (out_hw[i] != out_ref[i]) {
            if (errors < 10) {
                std::cout << "Mismatch @ " << i
                          << " HW=" << (int)out_hw[i]
                          << " REF=" << (int)out_ref[i] << "\n";
            }
            errors++;
        }
    }

    if (errors == 0)
        std::cout << "✅ TEST PASSED: Outputs match!\n";
    else
        std::cout << "❌ TEST FAILED: " << errors << " mismatches\n";

    // -------------------------------------------------
    // Cleanup
    // -------------------------------------------------
    delete[] in_fm;
    delete[] out_hw;
    delete[] out_ref;
    delete[] weights;
    delete[] bias;

    return (errors == 0) ? 0 : 1;
}

