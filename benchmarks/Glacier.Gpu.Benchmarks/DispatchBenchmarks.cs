using System;
using BenchmarkDotNet.Attributes;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.RingBuffer;

namespace Glacier.Gpu.Benchmarks;

[MemoryDiagnoser]
public class DispatchBenchmarks : IDisposable
{
    private NvidiaSassEngine? _engine;
    private PersistentRingBuffer? _ringBuffer;
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

        _ringBuffer = _engine.GetOrCreateRingBuffer();

        nuint bytes = (nuint)(N * sizeof(float));
        CuDriver.Check(CuDriver.MemAlloc(out _d_a, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out _d_b, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out _d_c, bytes), "cuMemAlloc");
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

    [GlobalCleanup]
    public void Cleanup()
    {
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
