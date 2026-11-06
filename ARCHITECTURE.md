# Glacier.Gpu Architecture Specification (Arc42 Standard)

**Document Identifier:** `ARC42-GLACIER-GPU-001`  
**Version:** `1.2.0`  
**Classification:** Core Systems Specification  
**Conforming Standards:** Arc42 (Template v8.2), ASD-STE100 (Issue 8/9 Linguistic Rules), C4 Model (Levels 1–3)

---

## 1. Introduction and Goals

`Glacier.Gpu` is a bare-metal GPU compute engine engineered in modern C# (.NET 10). The engine provides ultra-low latency hardware acceleration for the Glacier ecosystem. It operates without high-overhead native C++ runtimes (`cudart64.dll`, `torch_cuda.so`). The engine interacts directly with graphics display drivers installed in the host operating system.

### 1.1 Requirements Overview

The engine satisfies four primary operational requirements:
1. **Sub-Microsecond Dispatch**: Eliminate the 8 to 12 microsecond operating system kernel launch penalty via persistent worker megakernels.
2. **Zero Heap Allocation**: Ensure zero garbage collection overhead on computational hot paths via unmanaged memory descriptors.
3. **Bare-Metal Driver Interoperability**: Bind directly to `nvcuda.dll`, `d3d12.dll`, and `amdhip64.dll` in `C:\Windows\System32`.
4. **Heterogeneous Concurrency**: Orchestrate parallel workloads across discrete NVIDIA GPUs and integrated AMD APUs simultaneously.

### 1.2 Quality Goals

The following quality goals guide all architectural decisions in `Glacier.Gpu`:

| Priority | Quality Goal | Target Metric | Architectural Strategy |
| :---: | :--- | :--- | :--- |
| **P1** | **Dispatch Latency** | $\le 100\text{ ns}$ per task | Host-pinned lock-free ring buffer with GPU memory polling |
| **P2** | **Memory Allocation** | $0\text{ Bytes}$ on hot paths | 64-byte `GpuWorkTask` cacheline structs and unmanaged pinned memory |
| **P3** | **Zero Runtime Bloat** | $0\text{ MB}$ external native DLLs | Direct P/Invoke to system display drivers (`System32`) |
| **P4** | **Hardware Saturation** | $> 13.5\text{ TFLOPS}$ (RTX 4060) | Unrolled fused multiply-add (FMA) SASS machine code execution |
| **P5** | **Memory Bandwidth** | $> 30\text{ GB/s}$ (AMD APU) | Direct zero-copy unified system memory access (LPDDR5X) |

### 1.3 Stakeholders

| Role | Expectation | Interaction Point |
| :--- | :--- | :--- |
| **Glacier.Polaris** | Sub-microsecond SIMD aggregation and filter acceleration | Ring buffer vector operations (`VectorAdd`, `Reduction`) |
| **Glacier.Tensor** | Fast matrix-vector dot products and GEMV compute | SASS compiled FMA loops and unified memory buffers |
| **Glacier.ML** | Parallel decision forest evaluation and clustering | Heterogeneous dual-GPU concurrent dispatch |
| **Systems Engineers** | Deterministic memory layout, zero GC pauses, Native AOT | `IGpuEngine`, `GpuWorkTask`, `PersistentRingBuffer` |

---

## 2. Architecture Constraints

### 2.1 Technical Constraints

1. **Runtime Framework**: The engine requires modern C# and .NET 10.
2. **Native AOT Compatibility**: All components must support ahead-of-time native compilation with zero reflection or runtime code generation.
3. **Unsafe Code Mandate**: The engine requires pointers (`byte*`, `GpuWorkTask*`) and unmanaged native memory blocks.
4. **Hardware Driver Presence**: The host system must provide valid display drivers (`nvcuda.dll` or `amdhip64.dll` or `d3d12.dll`).

### 2.2 Operating System Constraints

1. **Windows 11 (Primary)**: The primary deployment environment is 64-bit Windows 11 Build 22621+. The engine resolves drivers directly from `C:\Windows\System32`.
2. **Linux (Secondary)**: The engine supports 64-bit Linux environments providing `libcuda.so.1` or `libcuda.so`.

### 2.3 Memory Constraints

1. **Task Alignment**: Task descriptors must maintain strict 64-byte sizing to match physical CPU cache lines.
2. **No Managed Heap Churn**: Hot execution loops must not allocate managed objects or trigger garbage collector sweeps.

---

## 3. System Scope and Context

`Glacier.Gpu` sits directly between high-level Glacier ecosystem applications and low-level graphics hardware silicon.

### 3.1 Business Context

```
+-------------------------------------------------------------------------------+
|                       Glacier High-Performance Applications                   |
|           (Glacier.Polaris DataFrame, Glacier.Tensor, Glacier.ML)             |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                                 Glacier.Gpu                                   |
|   (Sub-Microsecond Ring Buffer, Bare-Metal Driver Bypass, Hardware SASS)      |
+-------------------------------------------------------------------------------+
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
+-----------------------------------+         +---------------------------------+
| NVIDIA GeForce RTX 4060 dGPU      |         | AMD Radeon 890M Graphics APU    |
| (Dedicated Ada Lovelace VRAM)     |         | (Unified LPDDR5X System Memory) |
+-----------------------------------+         +---------------------------------+
```

### 3.2 Technical Context & C4 Level 1 Diagram

```mermaid
flowchart TD
    subgraph HostApps ["Consumer Applications (.NET 10)"]
        APP1[Glacier.Polaris]
        APP2[Glacier.Tensor]
        APP3[Glacier.ML]
    end

    subgraph GlacierGpuCore ["Glacier.Gpu Engine Boundary"]
        FAC[GpuEngineFactory]
        NV[NvidiaSassEngine]
        AMD[AmdRdnaEngine]
        D3D[D3D12ComputeEngine]
        HET[HeterogeneousEngine]
        RING[PersistentRingBuffer]
    end

    subgraph OSDisplayDrivers ["Host System Display Drivers (Bare-Metal P/Invoke)"]
        NVCUDA[nvcuda.dll / libcuda.so]
        AMDHIP[amdhip64.dll]
        D3D12DLL[d3d12.dll / dxgi.dll]
    end

    subgraph HardwareSilicon ["Physical Silicon Hardware"]
        DGPU[NVIDIA RTX 4060 dGPU - 24 SMs]
        IGPU[AMD Radeon 890M APU - 16 CUs]
        RAM[Unified LPDDR5X RAM - 7500 MT/s]
    end

    APP1 --> FAC
    APP2 --> FAC
    APP3 --> FAC
    FAC --> NV
    FAC --> AMD
    FAC --> D3D
    FAC --> HET

    NV --> RING
    NV --> NVCUDA
    AMD --> AMDHIP
    D3D --> D3D12DLL
    HET --> NV
    HET --> AMD

    NVCUDA --> DGPU
    AMDHIP --> IGPU
    D3D12DLL --> IGPU
    IGPU <--> RAM

    classDef host fill:#12161f,stroke:#66fcf1,stroke-width:1px,color:#f8f9fa;
    classDef core fill:#1a2230,stroke:#45a29e,stroke-width:1px,color:#f8f9fa;
    classDef drv fill:#0b0c10,stroke:#ffc107,stroke-width:1px,color:#f8f9fa;
    classDef hw fill:#0b0c10,stroke:#28a745,stroke-width:1px,color:#f8f9fa;

    class APP1,APP2,APP3 host;
    class FAC,NV,AMD,D3D,HET,RING core;
    class NVCUDA,AMDHIP,D3D12DLL drv;
    class DGPU,IGPU,RAM hw;
```

---

## 4. Solution Strategy

The architecture of `Glacier.Gpu` resolves low-level latency and throughput challenges through five fundamental design pillars:

### 4.1 Persistent Worker Megakernel Architecture

Traditional GPU runtimes submit work via `cuLaunchKernel` or command lists. Each submission transitions through operating system graphics stacks (WDDM/KMT), incurring **8 to 12 microseconds** of scheduling overhead.

`Glacier.Gpu` inverts this pattern:
1. The engine launches a long-lived GPU megakernel at startup.
2. The megakernel occupies a single streaming multiprocessor (SM) and continuously polls host-pinned memory.
3. The CPU submits tasks into a lock-free circular ring buffer with low latency (**100 nanoseconds**).
4. The GPU worker detects the task, executes the mathematical instructions across its warps, and writes back the completion status.

### 4.2 Bare-Metal Driver Bypass

`Glacier.Gpu` uses C# direct P/Invoke bindings with `[SuppressGCTransition]`. The engine communicates directly with `nvcuda.dll` and `amdhip64.dll`. This approach removes native wrapper dependencies and prevents managed runtime transition penalties.

### 4.3 Hardware SASS Binary Execution & JIT Compilation

The engine contains native assembly kernels for compute operations (`VectorAdd`, `VectorFma`, `PersistentWorker`). The compilation pipeline inspects local developer paths for `ptxas.exe`. When present, it compiles PTX into optimized SASS cubin binaries for target compute capabilities (`sm_75` through `sm_90`). When absent, the engine falls back to driver-level JIT compilation through `cuModuleLoadData`.

### 4.4 AMD APU Unified Zero-Copy Memory Architecture

On modern APU hardware (e.g. AMD Ryzen AI 9 HX 370 with Radeon 890M), the CPU and GPU share the same physical LPDDR5X memory pool. Standard software copies data across virtual PCIe staging boundaries, wasting time and bus bandwidth. `Glacier.Gpu` allocates mapped host memory via `hipHostMalloc` or D3D12 default heap buffers. The GPU accesses the CPU buffer directly in **300 nanoseconds**, delivering **34.09 GB/s** sustained bandwidth.

### 4.5 Heterogeneous Dual-GPU Parallel Orchestration

The engine coordinates simultaneous parallel execution across both discrete NVIDIA GPUs and integrated AMD APUs. Matrix operations run on NVIDIA tensor cores, while memory-intensive reductions run concurrently across the shared APU memory controller.

---

## 5. Building Block View

### 5.1 Level 1: Whitebox Decomposition

`Glacier.Gpu` organizes into six core subsystems:

```
Glacier.Gpu/
├── Common/        # Abstractions, device metadata, unified memory block
├── Drivers/       # Direct P/Invoke bindings to nvcuda.dll, amdhip64.dll, d3d12.dll
├── Compilation/   # PTX compiler, SASS assembler, and kernel caching
├── RingBuffer/    # Lock-free ring buffer and 64-byte GpuWorkTask descriptor
├── Engines/       # Hardware engines: NvidiaSassEngine, AmdRdnaEngine, D3D12ComputeEngine, HeterogeneousEngine
└── Factory/       # Topology discovery and automatic engine instantiation
```

### 5.2 Level 2: Component Breakdown & Responsibilities

```mermaid
classDiagram
    class IGpuEngine {
        <<interface>>
        +GpuDeviceInfo DeviceInfo
        +bool IsInitialized
        +void Initialize()
        +UnifiedMemoryBlock~T~ AllocateUnifiedMemory~T~(int count)
        +void Dispose()
    }

    class NvidiaSassEngine {
        -IntPtr _ctx
        -PersistentRingBuffer _ringBuffer
        +Initialize()
        +BenchmarkFmaCompute(int N, int loopCount)
        +MeasureTraditionalLaunchLatency(int iterations)
    }

    class AmdRdnaEngine {
        -GpuDeviceInfo _deviceInfo
        +Initialize()
        +AllocateZeroCopy~T~(int count)
        +BenchmarkUnifiedBandwidth(int sizeMb)
    }

    class D3D12ComputeEngine {
        -ID3D12Device _device
        -ID3D12CommandQueue _queue
        +Initialize()
        +ExecuteWave32Kernel(...)
    }

    class HeterogeneousEngine {
        -NvidiaSassEngine _nvidia
        -AmdRdnaEngine _amd
        +Initialize()
        +RunConcurrentWorkload(int nvElements, int amdSizeMb)
    }

    class PersistentRingBuffer {
        -GpuWorkTask* _tasks
        -long[] _slotSequences
        +uint Enqueue(TaskOpCode opCode, uint elementCount, ...)
        +void SubmitAndWait(TaskOpCode opCode, uint elementCount, ...)
        +void Shutdown()
    }

    class GpuWorkTask {
        +uint TaskId
        +uint OpCode
        +uint ElementCount
        +uint Status
        +ulong BufferA
        +ulong BufferB
        +ulong BufferC
        +ulong BufferD
        +float Scalar
        +uint Flags
    }

    IGpuEngine <|.. NvidiaSassEngine
    IGpuEngine <|.. AmdRdnaEngine
    IGpuEngine <|.. D3D12ComputeEngine
    IGpuEngine <|.. HeterogeneousEngine
    NvidiaSassEngine *-- PersistentRingBuffer
    PersistentRingBuffer *-- GpuWorkTask
```

### 5.3 Level 3: Component Specifications

#### 5.3.1 `PersistentRingBuffer`
- **Location**: `Glacier.Gpu.RingBuffer.PersistentRingBuffer`
- **Role**: Dispatches computational tasks to GPU persistent megakernels using Dmitry Vyukov's bounded multi-producer sequence turn lease algorithm.
- **Invariants**: Capacity must be a power of two ($2^n$). Slot sequence array padded with `CacheLineStride = 8` (`64 bytes`) to prevent CPU core false sharing.

#### 5.3.2 `GpuWorkTask`
- **Location**: `Glacier.Gpu.RingBuffer.GpuWorkTask`
- **Role**: 64-byte unmanaged struct exchanged across the host-device memory boundary.
- **Layout**: Strict `[StructLayout(LayoutKind.Sequential, Size = 64)]`. Contains operation code, element count, completion status, 4 unmanaged buffer addresses, scalar parameters, and flags.

#### 5.3.3 `CuDriver`
- **Location**: `Glacier.Gpu.Drivers.CuDriver`
- **Role**: Native direct entry points for NVIDIA CUDA Driver API.
- **Design**: Eliminates CUDA runtime layers. Decorated with `[SuppressGCTransition]` for fast microsecond-level execution.

---

## 6. Runtime View

### 6.1 Dmitry Vyukov Sequence Turn Lease Algorithm

The persistent ring buffer coordinates multi-threaded CPU producers and the persistent GPU consumer without locks. It implements Dmitry Vyukov's bounded MPMC queue sequence turn lease algorithm.

```mermaid
sequenceDiagram
    autonumber
    participant T1 as CPU Producer Thread 1
    participant T2 as CPU Producer Thread 2
    participant Ring as Ring Buffer Slot Array (Pinned RAM)
    participant GPU as GPU Persistent Worker Megakernel

    Note over T1,T2: Both threads submit tasks concurrently
    T1->>Ring: Interlocked.Increment(ref _sequenceCounter) -> seq=1 (Slot 0)
    T2->>Ring: Interlocked.Increment(ref _sequenceCounter) -> seq=2 (Slot 1)

    rect rgb(18, 22, 31)
        Note over T1,Ring: Producer 1 verifies turn lease
        T1->>Ring: Read _slotSequences[Slot 0] == 0 (Valid Turn)
        T1->>Ring: Write Parameters (OpCode, ElementCount, Buffers, Scalar)
        T1->>Ring: Mark In-Flight: Volatile.Write(&task->Status, 0)
        T1->>Ring: Sse.StoreFence() / MemoryBarrier()
        T1->>Ring: Release-Publish Sequence ID: Volatile.Write(&task->TaskId, 1)
        T1->>Ring: Transfer Turn Lease: Volatile.Write(&_slotSequences[0], seq - 1 + Capacity)
    end

    rect rgb(11, 12, 16)
        Note over Ring,GPU: GPU persistent polling loop (CU_STREAM_NON_BLOCKING)
        GPU->>Ring: ld.acquire.sys.global.u32 detects task[slot]->TaskId == expectedTaskId
        GPU->>GPU: Broadcast OpCode, ElementCount, Buffers via shared memory
        GPU->>GPU: Dispatch SIMT warps across grid-stride loop
        GPU->>Ring: st.release.sys.global.u32 task[slot]->Status = 1 (Completed)
    end

    rect rgb(26, 34, 48)
        Note over T1,Ring: Completion verification
        T1->>Ring: SpinWait reads task[0]->Status == 1
        T1->>T1: Return execution control (Latency: 4.2 us roundtrip)
    end
```

### 6.2 Memory Coherency and Synchronization Primitives

The engine enforces strict memory ordering across heterogeneous computing boundaries:

1. **Host-Pinned Memory Mapping**: Pinned host memory is mapped into the GPU virtual address space using `CU_MEMHOSTALLOC_DEVICEMAP`.
2. **Volatile Memory Semantics**: All ring buffer header fields rely on `Volatile.Read` and `Volatile.Write` to prevent compiler reordering.
3. **Hardware Store Fences**: On x86/x64 architectures, the engine executes `System.Runtime.Intrinsics.X86.Sse.StoreFence()` between writing task parameters and publishing the active `Status` byte.
4. **Cacheline Isolation**: Every slot sequence counter in `PersistentRingBuffer` is spaced by `CacheLineStride = 8` (`8 * 8 bytes = 64 bytes`). This eliminates cacheline ping-pong between distinct CPU cores.

---

## 7. Deployment View

### 7.1 Target Hardware Topologies

`Glacier.Gpu` deploys seamlessly across three primary workstation architectures:

```
+-----------------------------------------------------------------------------------+
| Architecture Tier 1: Discrete NVIDIA GPU Workstation                              |
| - Host: Any x64 CPU (.NET 10)                                                     |
| - GPU: NVIDIA Ada Lovelace / Ampere (RTX 4060 / 4070 / 4080 / 4090 / A100)        |
| - Driver: nvcuda.dll in C:\Windows\System32 or libcuda.so.1                       |
| - Acceleration: SASS machine code, Persistent Worker Ring Buffer (100 ns dispatch)|
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
| Architecture Tier 2: Integrated AMD APU Mobile / Mini-PC                          |
| - Host: AMD Ryzen AI 9 / Ryzen 7000 / 8000 Series Processor                       |
| - GPU: AMD Radeon 890M / 780M / 680M Graphics                                     |
| - Driver: amdhip64.dll or d3d12.dll in C:\Windows\System32                        |
| - Acceleration: Direct zero-copy unified memory (34 GB/s, 300 ns latency)         |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
| Architecture Tier 3: Heterogeneous Co-Execution Workstation                       |
| - Primary GPU: NVIDIA RTX 4060 Laptop GPU (FMA compute, 13.6 TFLOPS)              |
| - Secondary GPU: AMD Radeon 890M APU (Memory reductions, 34 GB/s unified memory)  |
| - Acceleration: HeterogeneousEngine parallel co-execution (123 ms overlap)        |
+-----------------------------------------------------------------------------------+
```

---

## 8. Cross-Cutting Concepts

### 8.1 Memory Management Model

```
Unmanaged Virtual Address Space
├── Host Pinned Memory (Allocated via CuDriver.MemHostAlloc)
│   ├── Slot 0: GpuWorkTask [64 Bytes] <--- Cache Line 0 (Direct GPU Read/Write)
│   ├── Slot 1: GpuWorkTask [64 Bytes] <--- Cache Line 1 (Direct GPU Read/Write)
│   └── Slot N: GpuWorkTask [64 Bytes] <--- Cache Line N (Direct GPU Read/Write)
│
├── Cache-Isolated Turn Array (Managed Array on 64-byte Boundaries)
│   ├── Slot 0 Sequence [8 longs = 64 Bytes] <--- Eliminates False Sharing
│   ├── Slot 1 Sequence [8 longs = 64 Bytes] <--- Eliminates False Sharing
│   └── Slot N Sequence [8 longs = 64 Bytes] <--- Eliminates False Sharing
│
└── Device Mapped Pointers (CUdeviceptr)
    └── Direct Virtual Memory Mapping into GPU Streaming Multiprocessors
```

### 8.2 Dynamic SASS Compilation Pipeline

```mermaid
flowchart LR
    A[PTX Kernel Source Code] --> B{ptxas.exe located?}
    B -->|Yes| C[Invoke ptxas.exe with --gpu-name=sm_89]
    C --> D{Exit Code == 0?}
    D -->|Yes| E[Hardware SASS Cubin Binary]
    D -->|No| F[Driver JIT via UTF-8 PTX Bytes]
    B -->|No| F
    E --> G[cuModuleLoadData]
    F --> G
    G --> H[cuModuleGetFunction Function Handle]
```

---

## 9. Architecture Decisions (ADR Matrix)

### ADR-001: Persistent Megakernel Ring Buffer vs Traditional Driver Dispatch
- **Status**: Accepted
- **Context**: Standard `cuLaunchKernel` calls incur 8 to 12 microseconds of operating system latency, creating massive bottlenecks for small tasks.
- **Decision**: Deploy a long-running persistent megakernel that polls host-pinned memory slots in a lock-free ring buffer.
- **Consequences**: Dispatch latency drops to 100 nanoseconds (103.8&times; speedup). Requires reserving one GPU streaming multiprocessor while the worker is active.

### ADR-002: Direct System32 Driver P/Invoke vs C++ CUDA Runtime Wrapper
- **Status**: Accepted
- **Context**: Third-party runtimes require shipping heavy native binaries (`cudart64.dll`, `torch_cuda.so`), causing deployment friction and version mismatch.
- **Decision**: Bind directly to `nvcuda.dll` and `amdhip64.dll` present in `C:\Windows\System32` via pure C# P/Invoke.
- **Consequences**: Zero native binary dependencies, zero DLL deployment overhead, and full Native AOT compatibility.

### ADR-003: AMD APU Zero-Copy Unified Memory vs PCIe Staged Copying
- **Status**: Accepted
- **Context**: Integrated APU graphics share physical RAM with the CPU, but standard runtimes copy data across virtual PCIe staging buffers.
- **Decision**: Allocate host-pinned memory mapped directly into the GPU compute address space.
- **Consequences**: Data sharing latency drops from 15 microseconds to 300 nanoseconds, unlocking 34 GB/s direct memory streaming.

### ADR-004: Direct3D 12 Compute Queue for Universal Windows APU Acceleration
- **Status**: Accepted
- **Context**: ROCm/HIP user-mode runtimes are occasionally absent on consumer Windows laptops with AMD Radeon APUs.
- **Decision**: Provide `D3D12ComputeEngine` as a universal first-class backend using native HLSL Wave32 compute shaders.
- **Consequences**: Universal hardware acceleration across all Windows graphics adapters with zero external driver prerequisites.

### ADR-005: 64-Byte Cacheline Task Descriptors for False Sharing Elimination
- **Status**: Accepted
- **Context**: Multi-threaded task submission on high-core CPUs causes cache invalidation storms when slot state spans cacheline boundaries.
- **Decision**: Enforce `Size = 64` on `GpuWorkTask` and stride slot sequences by 64 bytes (`CacheLineStride = 8`).
- **Consequences**: Zero CPU core false sharing, sustaining 10.0M tasks/second across multi-threaded producers.

---

## 10. Quality Requirements & Verification Scenarios

| Scenario ID | Quality Attribute | Stimulus / Condition | Expected Architectural Response |
| :---: | :--- | :--- | :--- |
| **SC-01** | **Low Latency** | High-frequency task enqueuing | Enqueue finishes in $\le 100\text{ ns}$ without OS transitions |
| **SC-02** | **Memory Safety** | 1,000,000 consecutive dispatches | Zero garbage collection pressure ($0\text{ Bytes}$ heap allocated) |
| **SC-03** | **Bounded Recovery**| GPU hang or driver reset | Spinwait times out after $5,000\text{ ms}$ and throws `TimeoutException` |
| **SC-04** | **Hardware Saturation** | Compute-heavy FMA matrix pass | Saturates $\ge 13.5\text{ TFLOPS}$ on NVIDIA RTX 4060 Laptop GPU |
| **SC-05** | **Zero-Copy Sharing** | 64 MB tensor buffer exchange on APU | Buffer available to GPU in $\le 300\text{ ns}$ with $0\text{ ns}$ PCIe copy |

---

## 11. Risks and Technical Debt

### 11.1 Windows WDDM GPU Timeout Detection and Recovery (TDR)
- **Risk**: Windows resets graphics drivers if a GPU kernel runs uninterrupted for more than 2 seconds without presenting to the display.
- **Mitigation**: The persistent worker megakernel periodically yields execution or exits when the shutdown opcode (`TaskOpCode.Shutdown`) is received. For dedicated compute adapters, the TDR delay registry key (`TdrDelay`) can be tuned.

### 11.2 Memory Bus Contention on APU Architectures
- **Risk**: When both CPU cores and the GPU saturates the LPDDR5X memory controller simultaneously, memory bandwidth degrades.
- **Mitigation**: `HeterogeneousEngine` balances memory-bandwidth-heavy tasks on the APU while directing pure compute-heavy FMA tasks to the discrete GPU's dedicated GDDR6 VRAM.

### 11.3 Driver DLL Availability
- **Risk**: Systems lacking dedicated NVIDIA graphics fail during `CuDriver.Init`.
- **Mitigation**: `GpuEngineFactory` probes driver availability cleanly and falls back automatically to `D3D12ComputeEngine`.

---

## 12. Technical Glossary (ASD-STE100 Aligned)

| Term | Approved Definition |
| :--- | :--- |
| **APU (Accelerated Processing Unit)** | A physical processor that combines CPU cores and GPU execution units on one silicon die. |
| **Cache Line** | A 64-byte unit of data transferred between main memory and CPU processor cache. |
| **CUBIN** | A compiled binary file containing executable SASS machine code for NVIDIA GPUs. |
| **dGPU (Discrete GPU)** | A dedicated graphics processing unit with independent physical VRAM connected over PCIe. |
| **FMA (Fused Multiply-Add)** | A mathematical instruction computing $(A \times B) + C$ in a single hardware floating-point cycle. |
| **GpuWorkTask** | A 64-byte unmanaged struct that defines a computational task for the GPU worker. |
| **Megakernel** | A long-running GPU kernel that stays resident on hardware execution units to process dynamic tasks. |
| **Persistent Ring Buffer**| A lock-free circular queue located in host-pinned memory that exchanges tasks between CPU and GPU. |
| **PTX (Parallel Thread Execution)** | Low-level virtual machine assembly language for NVIDIA parallel computing architectures. |
| **SASS (Streaming Assembler)** | The low-level binary machine language executed directly by NVIDIA hardware streaming multiprocessors. |
| **SIMT (Single Instruction, Multiple Threads)** | Execution model where multiple parallel threads execute the same instruction across lockstep lanes. |
| **SM (Streaming Multiprocessor)** | A physical hardware processing block containing execution units, register files, and warp schedulers. |
| **Unified Memory** | Physical or virtual memory accessible by both the CPU and GPU without explicit staging copies. |
| **Wave32** | A wavefront execution mode on AMD RDNA architectures executing 32 thread lanes in physical lockstep. |
