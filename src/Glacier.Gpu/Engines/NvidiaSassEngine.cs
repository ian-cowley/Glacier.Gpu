using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Gpu.Common;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.RingBuffer;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Hardware engine for NVIDIA GPUs (Ada Lovelace, Ampere, Turing).
/// Bypasses CUDA runtime to execute raw SASS machine code and persistent ring buffer workers.
/// </summary>
public sealed unsafe class NvidiaSassEngine : IGpuEngine
{
    private IntPtr _ctx;
    private int _device;
    private IntPtr _modAdd;
    private IntPtr _modFma;
    private IntPtr _modWorker;
    private IntPtr _fnVectorAdd;
    private IntPtr _fnVectorFma;
    private IntPtr _fnWorker;
    private PersistentRingBuffer? _ringBuffer;
    private IntPtr _hostTaskPtr;
    private IntPtr _devTaskPtr;
    private GCHandle _hWorkerParam;
    private GCHandle _hWorkerArray;
    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;
    public PersistentRingBuffer? RingBuffer => _ringBuffer;
    public IntPtr ContextHandle => _ctx;

    public void Initialize()
    {
        if (!CuDriver.IsAvailable())
            throw new PlatformNotSupportedException("NVIDIA CUDA driver (nvcuda.dll) is not available on this system.");

        CuDriver.Check(CuDriver.Init(0), "cuInit");
        CuDriver.Check(CuDriver.DeviceGetCount(out int count), "cuDeviceGetCount");
        if (count == 0)
            throw new InvalidOperationException("No CUDA-capable GPU devices found.");

        CuDriver.Check(CuDriver.DeviceGet(out int dev, 0), "cuDeviceGet");
        _device = dev;
        string devName = CuDriver.GetDeviceName(dev);
        string arch = CuDriver.GetComputeCapability(dev);
        int smCount = CuDriver.GetMultiprocessorCount(dev);
        CuDriver.Check(CuDriver.DeviceTotalMem(out nuint totalMem, dev), "cuDeviceTotalMem");

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: devName,
            DeviceType: GpuDeviceType.NvidiaDiscrete,
            Architecture: arch,
            ComputeUnitsOrSms: smCount,
            TotalMemoryBytes: (long)totalMem,
            IsUnifiedMemory: false,
            SupportsPersistentRingBuffer: true
        );

        CuDriver.Check(CuDriver.CtxCreate(out _ctx, 0, dev), "cuCtxCreate");
        CuDriver.CtxSetCurrent(_ctx);

        // Load or JIT compile kernels
        byte[] cubinAdd = KernelCache.GetOrCompile(arch, "VectorAdd", KernelSources.VectorAddPtx);
        byte[] cubinFma = KernelCache.GetOrCompile(arch, "VectorFma", KernelSources.VectorFmaPtx);
        byte[] cubinWorker = KernelCache.GetOrCompile(arch, "PersistentWorker", KernelSources.PersistentWorkerPtx);

        CuDriver.Check(CuDriver.ModuleLoadData(out _modAdd, cubinAdd), "cuModuleLoadData(VectorAdd)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVectorAdd, _modAdd, "vector_add"), "cuModuleGetFunction(vector_add)");

        CuDriver.Check(CuDriver.ModuleLoadData(out _modFma, cubinFma), "cuModuleLoadData(VectorFma)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVectorFma, _modFma, "vector_fma_benchmark"), "cuModuleGetFunction(vector_fma)");

        CuDriver.Check(CuDriver.ModuleLoadData(out _modWorker, cubinWorker), "cuModuleLoadData(PersistentWorker)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnWorker, _modWorker, "persistent_ring_worker"), "cuModuleGetFunction(persistent_worker)");

        _initialized = true;
    }

    /// <summary>
    /// Allocates host-pinned device-mapped memory block.
    /// </summary>
    public UnifiedMemoryBlock<T> AllocateUnifiedMemory<T>(int count) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuDriver.CtxSetCurrent(_ctx);

        nuint bytes = (nuint)(count * sizeof(T));
        CuDriver.Check(CuDriver.MemHostAlloc(
            out IntPtr hostPtr, 
            bytes, 
            CuDriver.CU_MEMHOSTALLOC_DEVICEMAP | CuDriver.CU_MEMHOSTALLOC_PORTABLE), 
            "cuMemHostAlloc");

        CuDriver.Check(CuDriver.MemHostGetDevicePointer(out IntPtr devPtr, hostPtr, 0), "cuMemHostGetDevicePointer");

        return new UnifiedMemoryBlock<T>(hostPtr, devPtr, count, p => CuDriver.MemFreeHost(p));
    }

    /// <summary>
    /// Measures traditional OS driver kernel launch latency via cuLaunchKernel (nvcuda.dll).
    /// Typically 8 - 12 microseconds due to OS kernel driver submission overhead.
    /// </summary>
    public double MeasureTraditionalLaunchLatency(int iterations = 100_000)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuDriver.CtxSetCurrent(_ctx);
        int N = 1024;
        nuint bytes = (nuint)(N * sizeof(float));

        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_a, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_b, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_c, bytes), "cuMemAlloc");

        IntPtr[] kernelParams = new IntPtr[4];
        GCHandle hParam0 = GCHandle.Alloc(d_a, GCHandleType.Pinned);
        GCHandle hParam1 = GCHandle.Alloc(d_b, GCHandleType.Pinned);
        GCHandle hParam2 = GCHandle.Alloc(d_c, GCHandleType.Pinned);
        GCHandle hParam3 = GCHandle.Alloc(N, GCHandleType.Pinned);

        kernelParams[0] = hParam0.AddrOfPinnedObject();
        kernelParams[1] = hParam1.AddrOfPinnedObject();
        kernelParams[2] = hParam2.AddrOfPinnedObject();
        kernelParams[3] = hParam3.AddrOfPinnedObject();
        GCHandle hParamsArray = GCHandle.Alloc(kernelParams, GCHandleType.Pinned);
        IntPtr pParams = hParamsArray.AddrOfPinnedObject();

        // Warmup
        CuDriver.LaunchKernel(_fnVectorAdd, 4, 1, 1, 256, 1, 1, 0, IntPtr.Zero, pParams, IntPtr.Zero);
        CuDriver.CtxSynchronize();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            CuDriver.LaunchKernel(_fnVectorAdd, 4, 1, 1, 256, 1, 1, 0, IntPtr.Zero, pParams, IntPtr.Zero);
        }
        CuDriver.CtxSynchronize();
        sw.Stop();

        hParam0.Free();
        hParam1.Free();
        hParam2.Free();
        hParam3.Free();
        hParamsArray.Free();
        CuDriver.MemFree(d_a);
        CuDriver.MemFree(d_b);
        CuDriver.MemFree(d_c);

        return sw.Elapsed.TotalMicroseconds / iterations;
    }

    /// <summary>
    /// Measures CPU-to-GPU fire-and-forget enqueue latency vs round-trip synchronous latency.
    /// </summary>
    /// <summary>
    /// Gets or initializes the persistent ring buffer worker on the GPU with up to 1024 threads.
    /// </summary>
    public PersistentRingBuffer GetOrCreateRingBuffer(int threadCount = 1024)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_ringBuffer != null && !_ringBuffer.IsDisposed)
            return _ringBuffer;

        CuDriver.CtxSetCurrent(_ctx);
        int maxThreads = CuDriver.GetMaxThreadsPerBlock(_device);
        int actualThreads = Math.Clamp(threadCount, 32, maxThreads);

        const int DefaultRingCapacity = 64;
        nuint taskBytes = (nuint)(sizeof(GpuWorkTask) * DefaultRingCapacity);
        CuDriver.Check(CuDriver.MemHostAlloc(
            out _hostTaskPtr,
            taskBytes,
            CuDriver.CU_MEMHOSTALLOC_DEVICEMAP | CuDriver.CU_MEMHOSTALLOC_PORTABLE),
            "cuMemHostAlloc");
        CuDriver.Check(CuDriver.MemHostGetDevicePointer(out _devTaskPtr, _hostTaskPtr, 0), "cuMemHostGetDevicePointer");

        _ringBuffer = new PersistentRingBuffer(_hostTaskPtr, _devTaskPtr, DefaultRingCapacity);

        GpuWorkTask* task = (GpuWorkTask*)_hostTaskPtr;
        task->TaskId = 0;
        task->OpCode = (uint)TaskOpCode.Nop;
        task->ElementCount = (uint)actualThreads;
        task->Status = 1;

        IntPtr devTask = _devTaskPtr;
        IntPtr* launchParams = stackalloc IntPtr[1];
        launchParams[0] = (IntPtr)(&devTask);

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnWorker,
            1, 1, 1,
            (uint)actualThreads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)launchParams,
            IntPtr.Zero), "LaunchKernel(PersistentWorker)");

        Thread.Sleep(2); // Let worker spin up
        return _ringBuffer;
    }

    /// <summary>
    /// Measures CPU-to-GPU fire-and-forget enqueue latency vs round-trip synchronous latency across arbitrary vector sizes.
    /// </summary>
    public (double dispatchUs, double roundTripUs, bool verified, float sampleVal) MeasureRingBufferLatency(int iterations = 10_000, int elementCount = 256)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuDriver.CtxSetCurrent(_ctx);
        int N = elementCount;
        nuint bufferBytes = (nuint)(N * sizeof(float));

        // Allocate data buffers
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_a, bufferBytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_b, bufferBytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_c, bufferBytes), "cuMemAlloc");

        float[] initA = new float[N];
        float[] initB = new float[N];
        Array.Fill(initA, 10.0f);
        Array.Fill(initB, 20.0f);

        fixed (float* pA = initA, pB = initB)
        {
            CuDriver.MemcpyHtoD(d_a, (IntPtr)pA, bufferBytes);
            CuDriver.MemcpyHtoD(d_b, (IntPtr)pB, bufferBytes);
        }

        PersistentRingBuffer ring = GetOrCreateRingBuffer(1024);
        GpuWorkTask* task = (GpuWorkTask*)ring.HostTaskPointer;
        task->ElementCount = (uint)N;
        task->BufferA = (ulong)d_a;
        task->BufferB = (ulong)d_b;
        task->BufferC = (ulong)d_c;

        // 1. Enqueue latency benchmark (CPU writing tasks to ring buffer)
        var swEnqueue = Stopwatch.StartNew();
        for (uint i = 1; i <= (uint)iterations; i++)
        {
            task->OpCode = (uint)TaskOpCode.VectorAdd;
            if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
                System.Runtime.Intrinsics.X86.Sse.StoreFence();
            else
                Thread.MemoryBarrier();
            Volatile.Write(ref task->TaskId, i);
        }
        swEnqueue.Stop();
        double dispatchUs = swEnqueue.Elapsed.TotalMicroseconds / iterations;

        // 2. Synchronous round-trip turnaround
        int roundTripIters = Math.Min(5000, iterations);
        var swRoundTrip = Stopwatch.StartNew();
        for (uint i = 1; i <= (uint)roundTripIters; i++)
        {
            task->OpCode = (uint)TaskOpCode.VectorAdd;
            task->Status = 0;
            if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
                System.Runtime.Intrinsics.X86.Sse.StoreFence();
            else
                Thread.MemoryBarrier();
            Volatile.Write(ref task->TaskId, 100_000 + i);

            while (Volatile.Read(ref task->Status) == 0)
            {
                Thread.SpinWait(1);
            }
        }
        swRoundTrip.Stop();
        double roundTripUs = swRoundTrip.Elapsed.TotalMicroseconds / roundTripIters;

        // Verify VectorFma as well: C = A * 2.0f + B = 10 * 2 + 20 = 40
        task->OpCode = (uint)TaskOpCode.VectorFma;
        task->Scalar = 2.0f;
        task->Status = 0;
        if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        else
            Thread.MemoryBarrier();
        Volatile.Write(ref task->TaskId, 200_001);

        while (Volatile.Read(ref task->Status) == 0)
        {
            Thread.SpinWait(1);
        }

        // Shutdown worker before copying back device memory (MemcpyDtoH synchronizes stream)
        task->OpCode = (uint)TaskOpCode.Shutdown;
        if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        else
            Thread.MemoryBarrier();
        Volatile.Write(ref task->TaskId, task->TaskId + 1);
        CuDriver.CtxSynchronize();

        // Verify result after worker kernel shutdown
        float[] result = new float[N];
        fixed (float* pR = result)
        {
            CuDriver.MemcpyDtoH((IntPtr)pR, d_c, bufferBytes);
        }
        float sampleVal = result[0];
        bool verified = Math.Abs(sampleVal - 40.0f) < 0.001f && Math.Abs(result[N - 1] - 40.0f) < 0.001f;
        if (N > 256)
        {
            verified = verified && Math.Abs(result[256] - 40.0f) < 0.001f && Math.Abs(result[N / 2] - 40.0f) < 0.001f;
        }

        // Cleanup
        CuDriver.MemFree(d_a);
        CuDriver.MemFree(d_b);
        CuDriver.MemFree(d_c);
        if (_hostTaskPtr != IntPtr.Zero)
        {
            CuDriver.MemFreeHost(_hostTaskPtr);
            _hostTaskPtr = IntPtr.Zero;
            _devTaskPtr = IntPtr.Zero;
        }
        _ringBuffer = null;

        return (dispatchUs, roundTripUs, verified, sampleVal);
    }

    /// <summary>
    /// Executes SASS FMA unrolled compute benchmark.
    /// </summary>
    public (double tflops, double latencyMs) BenchmarkFmaCompute(int N = 4_194_304, int loopCount = 200)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuDriver.CtxSetCurrent(_ctx);
        nuint bytes = (nuint)(N * sizeof(float));

        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_a, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_b, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_c, bytes), "cuMemAlloc");

        IntPtr[] kernelParams = new IntPtr[5];
        GCHandle hParam0 = GCHandle.Alloc(d_a, GCHandleType.Pinned);
        GCHandle hParam1 = GCHandle.Alloc(d_b, GCHandleType.Pinned);
        GCHandle hParam2 = GCHandle.Alloc(d_c, GCHandleType.Pinned);
        GCHandle hParam3 = GCHandle.Alloc(N, GCHandleType.Pinned);
        GCHandle hParam4 = GCHandle.Alloc(loopCount, GCHandleType.Pinned);

        kernelParams[0] = hParam0.AddrOfPinnedObject();
        kernelParams[1] = hParam1.AddrOfPinnedObject();
        kernelParams[2] = hParam2.AddrOfPinnedObject();
        kernelParams[3] = hParam3.AddrOfPinnedObject();
        kernelParams[4] = hParam4.AddrOfPinnedObject();
        GCHandle hArray = GCHandle.Alloc(kernelParams, GCHandleType.Pinned);

        // Warmup
        CuDriver.LaunchKernel(_fnVectorFma, (uint)(N / 256), 1, 1, 256, 1, 1, 0, IntPtr.Zero, hArray.AddrOfPinnedObject(), IntPtr.Zero);
        CuDriver.CtxSynchronize();

        int iters = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            CuDriver.LaunchKernel(_fnVectorFma, (uint)(N / 256), 1, 1, 256, 1, 1, 0, IntPtr.Zero, hArray.AddrOfPinnedObject(), IntPtr.Zero);
        }
        CuDriver.CtxSynchronize();
        sw.Stop();

        hParam0.Free();
        hParam1.Free();
        hParam2.Free();
        hParam3.Free();
        hParam4.Free();
        hArray.Free();

        CuDriver.MemFree(d_a);
        CuDriver.MemFree(d_b);
        CuDriver.MemFree(d_c);

        double totalSecs = sw.Elapsed.TotalSeconds;
        double avgMs = sw.Elapsed.TotalMilliseconds / iters;
        double fmaOpsPerThread = (double)loopCount * 8.0 * 2.0;
        double totalFlops = (double)N * fmaOpsPerThread * iters;
        double tflops = (totalFlops / totalSecs) / 1e12;

        return (tflops, avgMs);
    }

    public void Synchronize()
    {
        if (_ctx != IntPtr.Zero)
        {
            CuDriver.CtxSetCurrent(_ctx);
            CuDriver.CtxSynchronize();
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _ringBuffer?.Dispose();
            _ringBuffer = null;

            if (_hWorkerParam.IsAllocated) _hWorkerParam.Free();
            if (_hWorkerArray.IsAllocated) _hWorkerArray.Free();

            if (_ctx != IntPtr.Zero)
            {
                CuDriver.CtxDestroy(_ctx);
                _ctx = IntPtr.Zero;
            }
        }
    }
}
