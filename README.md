# FPGA-Based YOLOv4-Tiny Accelerator on AMD Kria KV260
---

## 📌 Project Overview

This project implements a hardware-accelerated YOLOv4-Tiny object detection system on the AMD Kria KV260 platform. The accelerator is developed using Vitis HLS and integrated using the Vitis kernel flow. The objective is to offload compute-intensive convolution operations from the ARM processor to the FPGA fabric to improve inference performance.

The system demonstrates measurable acceleration over a CPU-only baseline.

---

## 🧠 System Architecture

Processing Pipeline:

Input Frame  
→ Preprocessing  
→ FPGA Accelerator  
→ Postprocessing  
→ Output Display  

The FPGA accelerator leverages parallel computation and optimized memory buffering for improved throughput.

---

## 🖥 Hardware Platform

- Board: AMD Kria KV260  
- Device: xck26-sfvc784-2LV-c  
- Design Flow: Vitis Kernel Flow  
- Runtime: Xilinx Runtime (XRT)  
- Programming Model: OpenCL  

---

## ⚙ Accelerator Features

- HLS-based custom accelerator
- AXI4 Memory-Mapped Interface
- Loop Pipelining
- Parallel MAC Units
- Efficient BRAM Utilization
- INT8 Computation Support

---

## 📊 Resource Utilization

| Resource | Utilization |
|----------|------------|
| LUT      | 18% |
| Flip-Flops | 8% |
| BRAM     | 13% |
| DSP      | 13% |

The design maintains balanced FPGA resource usage with scalability potential.

---

## 🚀 Performance Evaluation

### 🔹 CPU Baseline

| Metric | Value |
|--------|-------|
| Average Latency | 850 ms |
| Average FPS | 1.18 FPS |

### 🔹 FPGA Accelerator

| Metric | Value |
|--------|-------|
| Average Latency | 368 ms |
| Average FPS | 2.7 FPS |

### 🔹 Speedup

Speedup Calculation:

Speedup = CPU Latency / FPGA Latency  
= 850 / 368  
≈ 2.31×

The FPGA implementation achieves a **2.31× speedup** over CPU execution.

---

# 📦 Repository Structure

```
Kria-YOLOv4-Tiny-FPGA-Accelerator/
│
├── hardware/
│   ├── hls/
│   ├── vivado_project/
│   └── README.md
│
├── software/
│   ├── host.cpp
│   └── README.md
│
├── cpu_baseline/
│
├── docs/
│   └── FPGA_YOLO_Accelerator_Final_Report.pdf
│
└── README.md
```

---

# 🛠 Installation & Setup

---

## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
## ░  REQUIREMENTS
## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

• AMD Kria KV260  
• Vitis 2023.x  
• Vivado  
• Xilinx Runtime (XRT)  
• OpenCL support  
• Ubuntu Linux  

---

## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
## ░  BUILD FPGA KERNEL
## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

1. Open Vitis HLS  
2. Synthesize the accelerator  
3. Export RTL Kernel (.xo)  
4. Link kernel to generate `.xclbin`  

---

## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
## ░  BUILD HOST APPLICATION
## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Compile host application:

```bash
g++ host.cpp -o host \
-I/opt/xilinx/xrt/include \
-L/opt/xilinx/xrt/lib \
-lOpenCL -pthread -lrt
```

---

## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
## ░  RUN APPLICATION
## ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

```bash
./host binary_container_1.xclbin
```

The console will display:

- FPGA Latency
- FPS
- Execution results

---

# 📘 Documentation

The detailed final project report is available in:

docs/FPGA_YOLO_Accelerator_Final_Report.pdf

---

# 🎯 Key Contributions

- FPGA-based CNN acceleration
- Hardware-software co-design
- Performance benchmarking against CPU baseline
- Efficient resource utilization
- Measurable speedup validation

---

# 🔮 Future Improvements

- Increase compute parallelism
- Optimize DDR bandwidth usage
- Explore lower precision quantization
- Full YOLO backbone acceleration

---

# 👥 Team Members

- Dharshan S
- Sandhyaa K  
- Dhamarai Kannan A  

Chennai Institute of Technology

---

# 📄 License

This project is developed for academic and research purposes.
