using System;
using BenchmarkDotNet.Attributes;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.RingBuffer;

namespace Glacier.Gpu.Benchmarks;

[MemoryDiagnoser]
public unsafe class DispatchBenchmarks : IDisposable
{
    private NvidiaSassEngine? _engine;
    private PersistentRingBuffer? _ringBuffer;
    private CudaGraphExecutor? _graphExecutor;
    private IntPtr _module;
    private IntPtr _fnAdd;
    private IntPtr _d_a;
    private IntPtr _d_b;
    private IntPtr _d_c;
    private const int N = 256;

    [GlobalSetup]
    public void Setup()
    {
        if (!CuDriver.IsAvailable()) return;

        _engine = new NvidiaSassEngine();
        _engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(_engine.ContextHandle), "cuCtxSetCurrent");

        _ringBuffer = _engine.GetOrCreateRingBuffer();

        nuint bytes = (nuint)(N * sizeof(float));
        CuDriver.Check(CuDriver.MemAlloc(out _d_a, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out _d_b, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out _d_c, bytes), "cuMemAlloc");

        byte[] cubin = KernelCache.GetOrCompile(_engine.DeviceInfo.Architecture, "VectorAdd", KernelSources.VectorAddPtx);
        CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), "cuModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAdd, _module, "vector_add"), "cuModuleGetFunction");

        _graphExecutor = new CudaGraphExecutor(_engine);
        _graphExecutor.TryCapture(stream =>
        {
            void** pParams = stackalloc void*[4];
            for (int k = 0; k < 16; k++)
            {
                IntPtr a = _d_a, b = _d_b, c = _d_c;
                int n = N;
                pParams[0] = &a;
                pParams[1] = &b;
                pParams[2] = &c;
                pParams[3] = &n;
                CuDriver.Check(CuDriver.LaunchKernel(_fnAdd, 1, 1, 1, 256, 1, 1, 0, stream, (IntPtr)pParams, IntPtr.Zero), "cuLaunchKernel");
            }
        });
    }

    [Benchmark(Baseline = true, Description = "Traditional cuLaunchKernel (Driver P/Invoke)")]
    public double TraditionalLaunch()
    {
        return _engine?.MeasureTraditionalLaunchLatency(1_000) ?? 0;
    }

    [Benchmark(Description = "Persistent Ring Buffer Enqueue (Direct Host-Pinned RAM)")]
    public uint RingBufferEnqueue()
    {
        if (_ringBuffer == null) return 0;
        return _ringBuffer.Enqueue(TaskOpCode.VectorAdd, (uint)N, (ulong)_d_a, (ulong)_d_b, (ulong)_d_c);
    }

    [Benchmark(Description = "CudaGraphExecutor 16-Kernel DAG Replay (Single cuGraphLaunch)")]
    public long CudaGraphReplay()
    {
        if (_graphExecutor == null) return 0;
        _graphExecutor.Replay(synchronize: true);
        return _graphExecutor.ReplayCount;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _graphExecutor?.Dispose();
        _graphExecutor = null;
        if (_module != IntPtr.Zero && _engine != null)
        {
            CuDriver.CtxSetCurrent(_engine.ContextHandle);
            CuDriver.ModuleUnload(_module);
            _module = IntPtr.Zero;
        }
        _ringBuffer?.Dispose();
        _ringBuffer = null;
        if (_d_a != IntPtr.Zero) { CuDriver.MemFree(_d_a); _d_a = IntPtr.Zero; }
        if (_d_b != IntPtr.Zero) { CuDriver.MemFree(_d_b); _d_b = IntPtr.Zero; }
        if (_d_c != IntPtr.Zero) { CuDriver.MemFree(_d_c); _d_c = IntPtr.Zero; }
        _engine?.Dispose();
        _engine = null;
    }

    public void Dispose() => Cleanup();
}
