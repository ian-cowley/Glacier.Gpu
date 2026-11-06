using System;
using System.Diagnostics;
using System.Threading;
using Glacier.Gpu.Common;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.RingBuffer;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Hardware engine for NVIDIA GPUs (Blackwell, Hopper, Ada Lovelace, Ampere, Turing).
/// Bypasses CUDA runtime to execute raw SASS machine code and multi-slot persistent ring buffer workers on dedicated non-blocking streams.
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
    private IntPtr _workerStream;
    private PersistentRingBuffer? _ringBuffer;
    private IntPtr _hostTaskPtr;
    private IntPtr _devTaskPtr;
    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;
    public PersistentRingBuffer? RingBuffer => _ringBuffer;
    public IntPtr ContextHandle => _ctx;

    public void Initialize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;

        if (!CuDriver.IsAvailable())
            throw new PlatformNotSupportedException("NVIDIA CUDA driver (nvcuda.dll / libcuda.so.1) is not available on this system.");

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
        byte[] cubinWorker = KernelCache.GetOrCompile(arch, "PersistentWorker_v2", KernelSources.PersistentWorkerPtx);

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

        return new UnifiedMemoryBlock<T>(hostPtr, devPtr, count, p =>
        {
            if (_ctx != IntPtr.Zero) CuDriver.CtxSetCurrent(_ctx);
            CuDriver.MemFreeHost(p);
        });
    }

    /// <summary>
    /// Measures traditional OS driver kernel launch latency via cuLaunchKernel (nvcuda.dll) with zero heap allocations.
    /// </summary>
    public double MeasureTraditionalLaunchLatency(int iterations = 100_000)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuDriver.CtxSetCurrent(_ctx);
        int n = 1024;
        nuint bytes = (nuint)(n * sizeof(float));

        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_a, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_b, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_c, bytes), "cuMemAlloc");

        try
        {
            void** pParams = stackalloc void*[4];
            pParams[0] = &d_a;
            pParams[1] = &d_b;
            pParams[2] = &d_c;
            pParams[3] = &n;

            // Warmup
            CuDriver.Check(CuDriver.LaunchKernel(_fnVectorAdd, 4, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pParams, IntPtr.Zero), "cuLaunchKernel(warmup)");
            CuDriver.StreamSynchronize(IntPtr.Zero);

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                CuDriver.LaunchKernel(_fnVectorAdd, 4, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pParams, IntPtr.Zero);
            }
            CuDriver.StreamSynchronize(IntPtr.Zero);
            sw.Stop();

            return sw.Elapsed.TotalMicroseconds / iterations;
        }
        finally
        {
            CuDriver.MemFree(d_a);
            CuDriver.MemFree(d_b);
            CuDriver.MemFree(d_c);
        }
    }

    /// <summary>
    /// Gets or initializes the multi-slot persistent ring buffer worker on a dedicated non-blocking CUDA stream.
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
        CuDriver.Check(CuDriver.StreamCreate(out _workerStream, CuDriver.CU_STREAM_NON_BLOCKING), "cuStreamCreate(non-blocking)");

        _ringBuffer = new PersistentRingBuffer(
            _hostTaskPtr,
            _devTaskPtr,
            DefaultRingCapacity,
            shutdownAction: () =>
            {
                if (_ctx != IntPtr.Zero && _workerStream != IntPtr.Zero)
                {
                    CuDriver.CtxSetCurrent(_ctx);
                    CuDriver.StreamSynchronize(_workerStream);
                }
            },
            disposeAction: () =>
            {
                if (_ctx != IntPtr.Zero)
                {
                    CuDriver.CtxSetCurrent(_ctx);
                    if (_workerStream != IntPtr.Zero)
                    {
                        CuDriver.StreamDestroy(_workerStream);
                        _workerStream = IntPtr.Zero;
                    }
                    if (_hostTaskPtr != IntPtr.Zero)
                    {
                        CuDriver.MemFreeHost(_hostTaskPtr);
                        _hostTaskPtr = IntPtr.Zero;
                        _devTaskPtr = IntPtr.Zero;
                    }
                }
                _ringBuffer = null;
            });

        IntPtr devTask = _devTaskPtr;
        uint ringMask = DefaultRingCapacity - 1;
        void** launchParams = stackalloc void*[2];
        launchParams[0] = &devTask;
        launchParams[1] = &ringMask;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnWorker,
            1, 1, 1,
            (uint)actualThreads, 1, 1,
            0, _workerStream,
            (IntPtr)launchParams,
            IntPtr.Zero), "LaunchKernel(PersistentWorker)");

        // Flush WDDM command queue immediately so the persistent worker starts executing on the SM without blocking CPU
        CuDriver.StreamQuery(_workerStream);

        // Warmup Nop task to confirm worker kernel is live on the SM before timing
        _ringBuffer.SubmitAndWait(TaskOpCode.Nop, 0, 0, 0, 0, 0f, timeoutMs: 5000);
        return _ringBuffer;
    }

    /// <summary>
    /// Measures genuine multi-slot PersistentRingBuffer Enqueue dispatch latency and SubmitAndWait round-trip latency across arbitrary vector sizes.
    /// </summary>
    public (double dispatchUs, double roundTripUs, bool verified, float sampleVal) MeasureRingBufferLatency(int iterations = 10_000, int elementCount = 256)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuDriver.CtxSetCurrent(_ctx);
        int n = Math.Max(1, elementCount);
        nuint bufferBytes = (nuint)(n * sizeof(float));

        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_a, bufferBytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_b, bufferBytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_c, bufferBytes), "cuMemAlloc");

        try
        {
            float[] initA = new float[n];
            float[] initB = new float[n];
            Array.Fill(initA, 10.0f);
            Array.Fill(initB, 20.0f);

            fixed (float* pA = initA, pB = initB)
            {
                CuDriver.Check(CuDriver.MemcpyHtoD(d_a, (IntPtr)pA, bufferBytes), "cuMemcpyHtoD(A)");
                CuDriver.Check(CuDriver.MemcpyHtoD(d_b, (IntPtr)pB, bufferBytes), "cuMemcpyHtoD(B)");
            }

            PersistentRingBuffer ring = GetOrCreateRingBuffer(1024);
            ulong bufA = (ulong)d_a;
            ulong bufB = (ulong)d_b;
            ulong bufC = (ulong)d_c;

            // 1. Genuine multi-slot Enqueue latency benchmark in burst batches of 32 slots
            int totalEnqueues = Math.Max(64, iterations);
            const int burstSize = 32;
            long enqueueTicks = 0;
            int dispatched = 0;
            while (dispatched < totalEnqueues)
            {
                int batch = Math.Min(burstSize, totalEnqueues - dispatched);
                long t0 = Stopwatch.GetTimestamp();
                for (int b = 0; b < batch; b++)
                {
                    ring.Enqueue(TaskOpCode.VectorAdd, (uint)n, bufA, bufB, bufC);
                }
                enqueueTicks += Stopwatch.GetTimestamp() - t0;
                dispatched += batch;

                // Drain in-flight burst before next timed burst so we measure pure fire-and-forget Enqueue latency
                ring.SubmitAndWait(TaskOpCode.Nop, 0, 0, 0, 0);
            }
            double dispatchUs = Stopwatch.GetElapsedTime(0, enqueueTicks).TotalMicroseconds / totalEnqueues;

            // 2. Genuine synchronous round-trip turnaround wrapping across all 64 slots
            int roundTripIters = Math.Clamp(iterations, 1, 5_000);
            var swRoundTrip = Stopwatch.StartNew();
            for (int i = 0; i < roundTripIters; i++)
            {
                ring.SubmitAndWait(TaskOpCode.VectorAdd, (uint)n, bufA, bufB, bufC);
            }
            swRoundTrip.Stop();
            double roundTripUs = swRoundTrip.Elapsed.TotalMicroseconds / roundTripIters;

            // 3. Verify VectorFma through PersistentRingBuffer: C = A * 2.0f + B = 10 * 2 + 20 = 40
            ring.SubmitAndWait(TaskOpCode.VectorFma, (uint)n, bufA, bufB, bufC, scalar: 2.0f);

            // Shut down persistent worker cleanly before synchronous cuMemcpyDtoH / cuMemFree
            // (cuMemFree implicitly synchronizes all active kernels in the CUDA context).
            ring.Dispose();

            float[] result = new float[n];
            fixed (float* pR = result)
            {
                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pR, d_c, bufferBytes), "cuMemcpyDtoH(C)");
            }

            float sampleVal = result[0];
            bool verified = Math.Abs(sampleVal - 40.0f) < 0.001f && Math.Abs(result[n - 1] - 40.0f) < 0.001f;
            if (n > 256)
            {
                verified = verified && Math.Abs(result[256] - 40.0f) < 0.001f && Math.Abs(result[n / 2] - 40.0f) < 0.001f;
            }

            return (dispatchUs, roundTripUs, verified, sampleVal);
        }
        finally
        {
            _ringBuffer?.Dispose();
            CuDriver.MemFree(d_a);
            CuDriver.MemFree(d_b);
            CuDriver.MemFree(d_c);
        }
    }

    /// <summary>
    /// Executes SASS FMA unrolled compute benchmark with zero heap allocations.
    /// </summary>
    public (double tflops, double latencyMs) BenchmarkFmaCompute(int N = 4_194_304, int loopCount = 200)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuDriver.CtxSetCurrent(_ctx);
        int n = N;
        int loops = loopCount;
        nuint bytes = (nuint)(n * sizeof(float));

        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_a, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_b, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_c, bytes), "cuMemAlloc");

        try
        {
            void** pParams = stackalloc void*[5];
            pParams[0] = &d_a;
            pParams[1] = &d_b;
            pParams[2] = &d_c;
            pParams[3] = &n;
            pParams[4] = &loops;

            uint gridX = (uint)((n + 255) / 256);

            // Warmup
            CuDriver.Check(CuDriver.LaunchKernel(_fnVectorFma, gridX, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pParams, IntPtr.Zero), "cuLaunchKernel(fma warmup)");
            CuDriver.StreamSynchronize(IntPtr.Zero);

            int iters = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                CuDriver.LaunchKernel(_fnVectorFma, gridX, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pParams, IntPtr.Zero);
            }
            CuDriver.StreamSynchronize(IntPtr.Zero);
            sw.Stop();

            double totalSecs = sw.Elapsed.TotalSeconds;
            double avgMs = sw.Elapsed.TotalMilliseconds / iters;
            double fmaOpsPerThread = (double)loops * 8.0 * 2.0;
            double totalFlops = (double)n * fmaOpsPerThread * iters;
            double tflops = (totalFlops / totalSecs) / 1e12;

            return (tflops, avgMs);
        }
        finally
        {
            CuDriver.MemFree(d_a);
            CuDriver.MemFree(d_b);
            CuDriver.MemFree(d_c);
        }
    }

    public void Synchronize()
    {
        if (_ctx != IntPtr.Zero && !_disposed)
        {
            CuDriver.CtxSetCurrent(_ctx);
            if (_ringBuffer != null && !_ringBuffer.IsDisposed)
            {
                CuDriver.StreamSynchronize(IntPtr.Zero);
            }
            else
            {
                CuDriver.CtxSynchronize();
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _initialized = false;
            _ringBuffer?.Dispose();
            _ringBuffer = null;

            if (_ctx != IntPtr.Zero)
            {
                CuDriver.CtxSetCurrent(_ctx);
                if (_workerStream != IntPtr.Zero)
                {
                    CuDriver.StreamDestroy(_workerStream);
                    _workerStream = IntPtr.Zero;
                }
                if (_hostTaskPtr != IntPtr.Zero)
                {
                    CuDriver.MemFreeHost(_hostTaskPtr);
                    _hostTaskPtr = IntPtr.Zero;
                    _devTaskPtr = IntPtr.Zero;
                }
                CuDriver.CtxDestroy(_ctx);
                _ctx = IntPtr.Zero;
            }
        }
    }
}
