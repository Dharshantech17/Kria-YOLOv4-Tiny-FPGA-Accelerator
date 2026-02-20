#include <ap_int.h>

typedef ap_int<8>  int8;
typedef ap_int<32> int32;

// =====================================================
// FINAL OPTIMIZED CONFIG (KV260 SAFE)
// =====================================================
#define H       26
#define W       26
#define CIN     128
#define COUT    128

#define K       3
#define CI_TILE 8      // optimized
#define CO_TILE 4

#define OH (H - K + 1)
#define OW (W - K + 1)

// =====================================================
// TOP FUNCTION (MATCH TCL)
// =====================================================
extern "C" {
void yolo_conv_kv260(
    int8  *in_fm,
    int8  *out_fm,
    int8  *weights,
    int32 *bias
) {
#pragma HLS INTERFACE m_axi port=in_fm   offset=slave bundle=gmem0 depth=86528
#pragma HLS INTERFACE m_axi port=out_fm  offset=slave bundle=gmem1 depth=86528
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=147456
#pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=gmem3 depth=128
#pragma HLS INTERFACE s_axilite port=return

    // =================================================
    // ON-CHIP BUFFERS
    // =================================================
    static int8 linebuf[K][W][CI_TILE];
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=3

    static int8 window[K][K][CI_TILE];
#pragma HLS ARRAY_PARTITION variable=window complete

    static int8 wbuf[CO_TILE][CI_TILE][K][K];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=2

    static int32 bias_buf[COUT];
#pragma HLS ARRAY_PARTITION variable=bias_buf cyclic factor=CO_TILE

    static int32 acc[CO_TILE];
#pragma HLS ARRAY_PARTITION variable=acc complete

    // =================================================
    // LOAD BIAS (ONCE)
    // =================================================
    for (int co = 0; co < COUT; co++) {
#pragma HLS PIPELINE
        bias_buf[co] = bias[co];
    }

    int row_ptr = 0;

    // =================================================
    // MAIN CONVOLUTION
    // =================================================
    for (int co0 = 0; co0 < COUT; co0 += CO_TILE) {
        for (int ci0 = 0; ci0 < CIN; ci0 += CI_TILE) {

            // ---------------------------------------------
            // LOAD WEIGHTS (PIPELINED)
            // ---------------------------------------------
            for (int co = 0; co < CO_TILE; co++)
                for (int ci = 0; ci < CI_TILE; ci++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++) {
#pragma HLS PIPELINE II=1
                            int idx =
                                (co0 + co) * CIN * K * K +
                                (ci0 + ci) * K * K +
                                ky * K + kx;
                            wbuf[co][ci][ky][kx] = weights[idx];
                        }

            row_ptr = 0;

            // ---------------------------------------------
            // SPATIAL LOOP (ROW-LEVEL PIPELINED)
            // ---------------------------------------------
            for (int y = 0; y < H; y++) {


                // Load one input row
                for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE
                    for (int ci = 0; ci < CI_TILE; ci++) {
                        int in_idx =
                            (y * W + x) * CIN +
                            (ci0 + ci);
                        linebuf[row_ptr][x][ci] = in_fm[in_idx];
                    }
                }

                // Compute once window is valid
                if (y >= K - 1) {
                    for (int x = K - 1; x < W; x++) {
#pragma HLS PIPELINE II=4

                        // Build sliding window (NO modulo on RAM)
                        for (int ky = 0; ky < K; ky++)
                            for (int kx = 0; kx < K; kx++)
                                for (int ci = 0; ci < CI_TILE; ci++) {
                                    int r = row_ptr + K - ky;
                                    if (r >= K) r -= K;
                                    window[ky][kx][ci] =
                                        linebuf[r][x - kx][ci];
                                }

                        // MAC computation
                        for (int co = 0; co < CO_TILE; co++) {
#pragma HLS UNROLL
                            if (ci0 == 0)
                                acc[co] = bias_buf[co0 + co];

                            for (int ci = 0; ci < CI_TILE; ci++) {
#pragma HLS UNROLL factor=2   // safe partial unroll
                                for (int ky = 0; ky < K; ky++)
                                    for (int kx = 0; kx < K; kx++)
                                        acc[co] +=
                                            window[ky][kx][ci] *
                                            wbuf[co][ci][ky][kx];
                            }

                            // Write output after last CI tile
                            if (ci0 == CIN - CI_TILE) {
                                int32 out = acc[co] >> 8;
                                if (out > 127)  out = 127;
                                if (out < -128) out = -128;

                                int out_idx =
                                    ((y - (K - 1)) * OW +
                                     (x - (K - 1))) * COUT +
                                    (co0 + co);

                                out_fm[out_idx] = (int8)out;
                            }
                        }
                    }
                }

                // Rotate line buffer row
                row_ptr++;
                if (row_ptr == K) row_ptr = 0;
            }
        }
    }
}
}

