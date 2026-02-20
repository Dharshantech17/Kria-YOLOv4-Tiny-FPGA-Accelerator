#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <cstring>
#include <iostream>
#include <chrono>

using namespace std;
using clk = chrono::high_resolution_clock;

/* ================= FPGA ================= */
#define FPGA_BASE  0xA0000000
#define MAP_SIZE   0x10000

#define REG_CTRL        0x00
#define REG_IN_FM_L     0x10
#define REG_IN_FM_H     0x14
#define REG_OUT_FM_L    0x1C
#define REG_OUT_FM_H    0x20
#define REG_WGT_L       0x28
#define REG_WGT_H       0x2C
#define REG_BIAS_L      0x34
#define REG_BIAS_H      0x38

volatile uint32_t* fpga;

/* ================= SIMPLE DDR BUFFERS ================= */
/* NOTE: This assumes identity mapping (common on KV260).
 * For production, use CMA or u-dma-buf.
 */

#define IN_SIZE    (26*26*128)
#define OUT_SIZE   (24*24*128)
#define WGT_SIZE   (128*128*3*3)
#define BIAS_SIZE  (128)

uint8_t  in_fm [IN_SIZE];
uint8_t  out_fm[OUT_SIZE];
uint8_t  weights[WGT_SIZE];
int32_t  bias[BIAS_SIZE];

/* ================= MAIN ================= */
int main() {

    /* -------- Init FPGA MMIO -------- */
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    fpga = (uint32_t*)mmap(
        nullptr, MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED, fd, FPGA_BASE
    );

    if (fpga == MAP_FAILED) {
        perror("mmap");
        return -1;
    }

    cout << "[INFO] FPGA MMIO mapped\n";

    /* -------- Init test data -------- */
    memset(in_fm, 1, IN_SIZE);
    memset(out_fm, 0, OUT_SIZE);
    memset(weights, 1, WGT_SIZE);
    for (int i = 0; i < BIAS_SIZE; i++)
        bias[i] = 0;

    uint64_t in_phys   = (uint64_t)in_fm;
    uint64_t out_phys  = (uint64_t)out_fm;
    uint64_t wgt_phys  = (uint64_t)weights;
    uint64_t bias_phys = (uint64_t)bias;

    /* -------- Write registers -------- */
    fpga[REG_IN_FM_L /4]  = (uint32_t)in_phys;
    fpga[REG_IN_FM_H /4]  = (uint32_t)(in_phys >> 32);
    fpga[REG_OUT_FM_L/4]  = (uint32_t)out_phys;
    fpga[REG_OUT_FM_H/4]  = (uint32_t)(out_phys >> 32);
    fpga[REG_WGT_L   /4]  = (uint32_t)wgt_phys;
    fpga[REG_WGT_H   /4]  = (uint32_t)(wgt_phys >> 32);
    fpga[REG_BIAS_L  /4]  = (uint32_t)bias_phys;
    fpga[REG_BIAS_H  /4]  = (uint32_t)(bias_phys >> 32);

    cout << "[INFO] Registers written\n";

    /* -------- Run FPGA -------- */
    auto t0 = clk::now();

    fpga[REG_CTRL /4] = 1;          // ap_start
    while ((fpga[REG_CTRL /4] & 0x2) == 0); // wait ap_done

    auto t1 = clk::now();

    double ms =
        chrono::duration<double, milli>(t1 - t0).count();

    cout << "[RESULT] FPGA convolution time: "
         << ms << " ms\n";

    /* -------- Sanity check -------- */
    cout << "[CHECK] Output[0..7]: ";
    for (int i = 0; i < 8; i++)
        cout << int(out_fm[i]) << " ";
    cout << endl;

    return 0;
}