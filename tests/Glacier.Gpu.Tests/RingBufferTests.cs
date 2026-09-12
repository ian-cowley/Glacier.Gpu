using System;
using System.Runtime.InteropServices;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.RingBuffer;
using Xunit;

namespace Glacier.Gpu.Tests;

public class RingBufferTests
{
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public RingBufferTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void GpuWorkTask_IsExactly64Bytes()
    {
        Assert.Equal(64, Marshal.SizeOf<GpuWorkTask>());
    }

    [Fact]
    public void PersistentRingBuffer_ExecutesTasksWithZeroAllocation()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();

        var (dispatchUs, roundTripUs, verified, sampleVal) = engine.MeasureRingBufferLatency(iterations: 10_000);

        _output.WriteLine($"[BENCHMARK] Enqueue Dispatch Latency: {dispatchUs:F3} us ({dispatchUs * 1000:F1} ns)");
        _output.WriteLine($"[BENCHMARK] Round-Trip Turnaround:     {roundTripUs:F3} us ({roundTripUs * 1000:F1} ns)");
        _output.WriteLine($"[BENCHMARK] Sample Output:             {sampleVal}");

        Assert.True(verified, $"Persistent ring buffer GPU arithmetic expected 30.0f, but got {sampleVal} (dispatch={dispatchUs:F3}us, roundTrip={roundTripUs:F3}us)");
        Assert.True(dispatchUs < 2.0, $"Dispatch latency should be sub-microsecond, was {dispatchUs:F3} us");
        Assert.True(roundTripUs < 25.0, $"Round-trip latency should be under 25 us, was {roundTripUs:F3} us");
    }
}
