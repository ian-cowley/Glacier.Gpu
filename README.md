![Glacier.Gpu Banner](assets/banner.jpg)

# Glacier.Gpu

**Heterogeneous Bare-Metal GPU Compute Engine in C# .NET 10**  
*Direct Driver APIs &bull; Raw SASS Machine Code &bull; Sub-Microsecond Ring Buffer &bull; Zero-Copy AMD APU &bull; Dual-GPU Orchestration*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Native AOT](https://img.shields.io/badge/Native%20AOT-Ready-brightgreen.svg)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)
[![NuGet Version](https://img.shields.io/nuget/v/Glacier.Gpu.svg)](https://www.nuget.org/packages/Glacier.Gpu/)
[![Ecosystem](https://img.shields.io/badge/Glacier-Ecosystem-blue)](https://github.com/ian-cowley)
[![Hardware: Bare-Metal GPU](https://img.shields.io/badge/Hardware-Bare--Metal%20GPU-brightgreen.svg)](#)
[![Website](https://img.shields.io/badge/Website-glaciersoftware.dev-66fcf1.svg)](https://glaciersoftware.dev/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-orange.svg?logo=buy-me-a-coffee)](https://buymeacoffee.com/glaciersoftware)

---

## 1. Overview

`Glacier.Gpu` is a bare-metal GPU compute engine for modern C# and .NET 10. The engine provides low-latency acceleration for the Glacier ecosystem. It accelerates columnar operations in `Glacier.Polaris` and deep learning tensor kernels in `Glacier.Tensor`.

Standard GPU libraries rely on heavy native C++ wrappers such as `cudart64.dll`, `torch_cuda.so`, and external runtimes. In contrast, `Glacier.Gpu` communicates directly with display drivers installed in the host operating system. The engine binds directly to `nvcuda.dll`, `d3d12.dll`, and `amdhip64.dll` in `C:\Windows\System32` and `libcuda.so.1` on Linux.

> [!NOTE]
> The engine requires no CUDA Toolkit installation and no external native C++ runtime binaries. It runs on clean consumer machines through direct operating system driver bindings.

```
+-------------------------------------------------------------------------------+
|                       Glacier High-Performance Applications                   |
|                 (Glacier.Polaris, Glacier.Tensor, Glacier.ML)                |
+-------------------------------------------------------------------------------+
                                      |
+-------------------------------------------------------------------------------+
|                                Glacier.Gpu                                    |
|  +-----------------------------------+  +----------------------------------+  |
|  |     NvidiaSassEngine (dGPU)       |  |       AmdRdnaEngine (APU)        |  |
|  | - Persistent Megakernel Ring      |  | - Direct Zero-Copy Host Memory   |  |
|  | - 100 ns Task Dispatch Latency    |  | - 300 ns RAM Sharing Latency     |  |
|  | - Ada Lovelace SASS (13.6 TFLOPS) |  | - 34 GB/s Unified Bandwidth      |  |
|  +-----------------------------------+  +----------------------------------+  |
|                                      |                                        |
|  +-------------------------------------------------------------------------+  |
|  |                     HeterogeneousEngine Coordinator                     |  |
|  |     Simultaneous Dual-GPU Parallel Co-Execution (dGPU + APU)            |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
+------------------------------------+    +------------------------------------+
| NVIDIA Display Driver (nvcuda.dll) |    |   AMD Driver Runtime (amdhip64.dll)|
| Windows System32 Direct Driver API |    | Windows System32 Direct Driver API |
+------------------------------------+    +------------------------------------+
```

---

## 2. Conceptual Architecture & Dual-Layer Diagram

### The Laymen Metaphor: "The Air Traffic Controller and Persistent Jet Patrol"

Standard GPU runtimes operate like a slow flight clearance office. Every time the CPU requests mathematical calculations, it dials the operating system driver. The driver verifies permissions, wakes the hardware scheduler, and launches an aircraft from the hangar. This handshake takes 8 to 12 microseconds. For micro-tasks, the processor spends 99% of its time waiting on flight clearance.

`Glacier.Gpu` changes this operational paradigm:

1. **Persistent Jet Patrol**: At startup, the engine launches a permanent squad of supersonic jets that hover continuously in formation over the runway (the Persistent GPU Worker Megakernel).
2. **Pneumatic Tube Dispatch**: The ground crew (host C# code) writes micro-task instructions into a 64-byte pneumatic capsule (the `GpuWorkTask` cacheline slot).
3. **Sub-Microsecond Interception**: The hovering jets read the capsule in **100 nanoseconds** without contacting the air traffic tower. They complete the computation at 13.6 trillion floating-point operations per second.
4. **Shared Airport Runway**: On modern APU processors, the CPU and GPU share the exact same physical DDR5 memory pool. Data transfers across the runway in **300 nanoseconds** with zero transport truck delays.

### Scientific Architecture Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant CPU as Host C# CPU Thread (.NET 10)
    participant Ring as Lock-Free Ring Buffer (Host-Pinned RAM)
    participant SM as NVIDIA GPU Streaming Multiprocessors (SMs)
    participant APU as AMD Radeon APU (Unified System RAM)

    Note over SM: Persistent Worker Megakernel Launches Once at Startup
    SM->>Ring: Continuous Memory Polling on volatile uint Status
    
    rect rgb(18, 22, 31)
        Note over CPU,Ring: Sub-Microsecond Task Submission (~100 ns)
        CPU->>Ring: Interlocked.Increment(ref _sequenceCounter)
        CPU->>Ring: Await Slot Turn Lease (Dmitry Vyukov Algorithm)
        CPU->>Ring: Write GpuWorkTask (64-Byte Cacheline Aligned Descriptor)
        CPU->>Ring: Set task->Status = 0 (Submitted/Running)
    end

    rect rgb(11, 12, 16)
        Note over Ring,SM: Zero-Driver-Overhead GPU Work Consumption
        SM->>Ring: Poll detects task->Status == 0
        SM->>SM: Decode TaskOpCode (VectorAdd, FMA, Gemv, Reduction)
        SM->>SM: Execute Warp Lockstep Instructions (32 Threads per Warp)
        SM->>Ring: Write task->Status = 1 (Completed)
    end

    rect rgb(26, 34, 48)
        Note over CPU,APU: Heterogeneous Concurrent Co-Execution (dGPU + APU)
        par NVIDIA dGPU (Ada Lovelace sm_89)
            CPU->>SM: Dispatch Compute-Dense FP32 FMA SASS Matrix Loops (13.6 TFLOPS)
        and AMD Radeon APU (RDNA 3.5 gfx1150)
            CPU->>APU: Stream High-Bandwidth Reductions over Zero-Copy RAM (34 GB/s)
        end
    end
    
    CPU->>Ring: SpinWait detects task->Status == 1 (Turnaround: 4.2 us)
```

### Foundational Scientific Literature & Citations

1. **GPU Architecture & SIMT Computing**: Lindholm, E., Nickolls, J., Oberman, S., & Montrym, J. (2008). *NVIDIA GeForce 8800 GPU Architecture*. IEEE Micro, 28(2), 39–55. [DOI: 10.1109/MM.2008.31](https://doi.org/10.1109/MM.2008.31).
2. **Massively Parallel Microprocessor Paradigms**: Nickolls, J., & Dally, W. J. (2010). *The GPU Computing Era*. IEEE Micro, 30(2), 56–69. [DOI: 10.1109/MM.2010.41](https://doi.org/10.1109/MM.2010.41).
3. **Lock-Free Bounded Ring Buffers**: Vyukov, D. (2014). *Bounded MPMC Queue Algorithm*. 1024cores.net. [Vyukov Queue Specification](https://web.archive.org/web/20230601000000*/https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue).
4. **Data-Level Parallelism and Memory Hierarchies**: Hennessy, J. L., & Patterson, D. A. (2017). *Computer Architecture: A Quantitative Approach* (6th ed.). Morgan Kaufmann. Chapter 4.
5. **AMD RDNA Unified Architecture**: AMD Corporation (2024). *AMD RDNA 3 / 3.5 Instruction Set Architecture and Unified Memory Architecture Whitepaper*.

---

## 3. Verified Hardware Benchmarks

All metrics reflect physical hardware execution on an **ASUS TUF Gaming A14 (FA608WV)** laptop workstation under Windows 11 Build 26100 and .NET 10.0 (Release x64 configuration).

- **Host Processor**: AMD Ryzen AI 9 HX 370 (12 Cores, 24 Threads, 36 MB Cache)
- **Discrete GPU (dGPU)**: NVIDIA GeForce RTX 4060 Laptop GPU (24 SMs, Ada Lovelace `sm_89`, 8 GB GDDR6)
- **Integrated GPU (APU)**: AMD Radeon 890M Graphics (16 CUs, RDNA 3.5 `gfx1150`, Shared LPDDR5X-7500)

| Benchmark Scenario | Traditional Baseline | Glacier.Gpu Engine | Measured Speedup / Throughput |
| :--- | :--- | :--- | :--- |
| **GPU Task Dispatch Latency** | 10.382 &mu;s (`cuLaunchKernel`) | **0.100 &mu;s (100.0 ns)** | **103.8&times; Lower Dispatch Latency** |
| **Task Dispatch Throughput** | 96,318 tasks/sec | **10,001,400 tasks/sec** | **10.0M tasks/sec Sustained** |
| **Turnaround Latency (Round-Trip)**| ~25.0 &mu;s (Stream sync) | **4.239 &mu;s (4.2 &mu;s)** | **5.9&times; Faster Total Turnaround** |
| **RTX 4060 SASS FP32 Compute** | Baseline Runtime API | **13.64 TFLOPS** | **Hardware Compute Saturation** |
| **AMD 890M Memory Sharing Latency** | PCIe Staging Buffer (~15 &mu;s) | **0.300 &mu;s (300 ns)** | **50&times; Lower Sharing Latency** |
| **AMD APU Unified Memory Bandwidth**| Host-to-Device Copy (~8 GB/s) | **34.09 GB/s** | **4.2&times; Higher Memory Bandwidth** |
| **Dual-GPU Concurrent Execution** | Sequential Dispatch (350 ms) | **123.28 ms** | **2.8&times; Parallel Workload Overlap** |
| **Wavefront Execution Tuning** | Wave64 Divergence Stalls | **Wave32 Dual-Issue (RDNA)** | **+33.8% Kernel Throughput Boost** |
| **LLM KV-Cache VRAM (8B at 32K)** | 8.00 GB FP32 (Crashes OOM) | **2.00 GB (Native FP8)** | **75% VRAM Reduction (4&times; Context Window)** |

---

## 4. Quickstart Tutorial

### Step 1: Install Glacier.Gpu

Add the package reference to your .NET 10 project file:

```bash
dotnet add package Glacier.Gpu
```

### Step 2: Initialize Hardware Engine and Verify Topology

Create the optimal hardware engine using automatic device discovery:

```csharp
using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Factory;

// Discover hardware and initialize the optimal GPU engine
using IGpuEngine engine = GpuEngineFactory.CreateOptimalEngine();

Console.WriteLine($"Initialized GPU Device: {engine.DeviceInfo.DeviceName}");
Console.WriteLine($"Architecture:           {engine.DeviceInfo.Architecture}");
Console.WriteLine($"Compute Units / SMs:    {engine.DeviceInfo.ComputeUnitsOrSms}");
Console.WriteLine($"Unified Memory:         {engine.DeviceInfo.IsUnifiedMemory}");
```

---

## 5. How-To Guides

### How-To Guide 1: Submit Tasks to the Persistent Ring Buffer

Submit micro-tasks to the GPU without operating system driver launch overhead:

```csharp
using System;
using Glacier.Gpu.Engines;
using Glacier.Gpu.RingBuffer;

using var nvidia = new NvidiaSassEngine();
nvidia.Initialize();

// Launch the 64-slot persistent ring buffer worker on a dedicated non-blocking CUDA stream
PersistentRingBuffer ringBuffer = nvidia.GetOrCreateRingBuffer();

// Declare device buffer addresses
ulong bufA = 0x1000, bufB = 0x2000, bufC = 0x3000;

// Enqueue tasks into the circular ring buffer with sub-microsecond dispatch latency
for (uint i = 0; i < 32; i++)
{
    ringBuffer.Enqueue(TaskOpCode.VectorAdd, elementCount: 256, bufA, bufB, bufC);
}

// Submit a task and wait for GPU completion with bounded spinwait
ringBuffer.SubmitAndWait(TaskOpCode.VectorAdd, elementCount: 256, bufA, bufB, bufC);
```

### How-To Guide 2: Allocate Zero-Copy Memory on AMD APUs

Share memory buffers directly between C# code and AMD GPU compute shaders:

```csharp
using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Engines;

using var amd = new AmdRdnaEngine();
amd.Initialize();

// Allocate 64 MB of unified memory accessible by both CPU and GPU
using UnifiedMemoryBlock<float> unified = amd.AllocateZeroCopy<float>(16_777_216);

// The CPU populates data directly through a C# Span<T> with zero copy overhead
Span<float> hostSpan = unified.Span;
hostSpan.Fill(42.0f);

// The GPU immediately consumes the buffer via unified.DevicePointer (0 ns PCIe latency)
Console.WriteLine($"Mapped Unified Device Address: 0x{unified.DevicePointer:X16}");
```

### How-To Guide 3: Execute Workloads Concurrently on Dual GPUs

Orchestrate simultaneous parallel execution across NVIDIA dGPU and AMD APU:

```csharp
using System;
using Glacier.Gpu.Engines;

using var het = new HeterogeneousEngine();
het.Initialize();

// Run heavy FP32 matrix compute on NVIDIA dGPU while concurrently sweeping memory on AMD APU
var (nvidiaTflops, amdBandwidth, totalMs) = het.RunConcurrentWorkload(
    nvElements: 4_194_304, 
    amdSizeMb: 256
);

Console.WriteLine($"Dual-GPU Workload Complete in {totalMs:F2} ms");
Console.WriteLine($"NVIDIA SASS FP32 Throughput: {nvidiaTflops:F2} TFLOPS");
Console.WriteLine($"AMD APU Unified Bandwidth:   {amdBandwidth:F2} GB/s");
```

---

## 6. Technical Reference & Architecture

For in-depth architectural specifications and comprehensive operational recipes, inspect:

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Arc42 12-section architecture specification, detailing bare-metal driver bypass, memory coherency, Dmitry Vyukov turn lease algorithms, and SASS compilation.
- **[USAGE.md](USAGE.md)**: Operational guide containing complete recipes for hardware discovery, PTX JIT compilation, Direct3D 12 execution, and troubleshooting.

### Core API Catalog

| Class / Struct | Namespace | Description |
| :--- | :--- | :--- |
| `GpuEngineFactory` | `Glacier.Gpu.Factory` | Probes system hardware and creates optimal GPU engine instances. |
| `NvidiaSassEngine` | `Glacier.Gpu.Engines` | Executes raw SASS machine code and persistent ring workers via `nvcuda.dll`. |
| `AmdRdnaEngine` | `Glacier.Gpu.Engines` | Manages zero-copy unified system memory access on AMD APUs via `amdhip64.dll`. |
| `D3D12ComputeEngine` | `Glacier.Gpu.Engines` | Drives compute queues on Windows display adapters with HLSL Wave32 compute shaders. |
| `HeterogeneousEngine`| `Glacier.Gpu.Engines` | Coordinates simultaneous parallel co-execution across discrete and integrated GPUs. |
| `PersistentRingBuffer` | `Glacier.Gpu.RingBuffer` | Lock-free bounded multi-slot circular ring buffer for persistent GPU workers. |
| `GpuWorkTask` | `Glacier.Gpu.RingBuffer` | 64-byte cacheline aligned task descriptor struct exchanged between CPU and GPU. |
| `UnifiedMemoryBlock<T>`| `Glacier.Gpu.Common` | Unmanaged pinned memory container exposing host pointers, device pointers, and `Span<T>`. |

---

## 7. Building and Testing

### Build Solution

Compile the project in Release configuration:

```bash
dotnet build Glacier.Gpu.slnx -c Release
```

### Run Unit Tests

Execute the automated test suite:

```bash
dotnet test Glacier.Gpu.slnx -c Release
```

### Run Real-World Hardware Benchmarks

Execute the high-resolution hardware benchmark suite:

```bash
dotnet run --project benchmarks/Glacier.Gpu.Benchmarks/Glacier.Gpu.Benchmarks.csproj -c Release
```

---

## 8. License & Ecosystem

Distributed under the **MIT License**. See [LICENSE](LICENSE) for full details.

Part of the **[Glacier Ecosystem](https://glaciersoftware.dev/)** — extreme performance, zero-allocation developer tooling and data engines natively engineered for modern C# and .NET 10.

* 🌐 **Documentation Hub**: [glaciersoftware.dev](https://glaciersoftware.dev/)
* ☕ **Buy Me a Coffee**: [buymeacoffee.com/glaciersoftware](https://buymeacoffee.com/glaciersoftware)
* ❤️ **GitHub Sponsors**: [github.com/sponsors/ian-cowley](https://github.com/sponsors/ian-cowley)
* 🐙 **GitHub Organization**: [github.com/ian-cowley](https://github.com/ian-cowley)
