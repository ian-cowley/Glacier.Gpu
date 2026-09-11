using System;
using System.Runtime.InteropServices;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.RingBuffer;
using Xunit;

namespace Glacier.Gpu.Tests;

public class RingBufferTests
{
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

        Assert.True(verified, $"Persistent ring buffer GPU arithmetic expected 30.0f, but got {sampleVal} (dispatch={dispatchUs:F3}us, roundTrip={roundTripUs:F3}us)");
        Assert.True(dispatchUs < 2.0, $"Dispatch latency should be sub-microsecond, was {dispatchUs:F3} us");
        Assert.True(roundTripUs < 25.0, $"Round-trip latency should be under 25 us, was {roundTripUs:F3} us");
    }
}
