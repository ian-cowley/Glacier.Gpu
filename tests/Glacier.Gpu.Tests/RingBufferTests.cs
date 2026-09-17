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

    [Theory]
    [InlineData(100_000)]
    [InlineData(1_000_000)]
    public void PersistentRingBuffer_GridStrideLoop_ComputesLargeVectorsCorrectly(int elementCount)
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();

        var (dispatchUs, roundTripUs, verified, sampleVal) = engine.MeasureRingBufferLatency(iterations: 100, elementCount: elementCount);

        _output.WriteLine($"[LARGE VECTOR N={elementCount:N0}] Round-Trip: {roundTripUs:F3} us | Sample: {sampleVal} | Verified: {verified}");

        Assert.True(verified, $"Persistent ring buffer grid-stride failed on vector size N={elementCount}. Truncation detected!");
        Assert.Equal(40.0f, sampleVal, precision: 3);
    }

    [Fact]
    public void D3D12_PersistentRingBuffer_ExecutesTasksWithZeroAllocation()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var engine = new D3D12ComputeEngine();
        engine.Initialize();

        Assert.True(engine.DeviceInfo.SupportsPersistentRingBuffer);

        var (dispatchUs, roundTripUs, verified, sampleVal) = engine.MeasureRingBufferLatency(iterations: 10_000, elementCount: 256);

        _output.WriteLine($"[D3D12 BENCHMARK] Adapter: {engine.DeviceInfo.DeviceName} ({engine.DeviceInfo.Architecture})");
        _output.WriteLine($"[D3D12 BENCHMARK] Enqueue Dispatch Latency: {dispatchUs:F3} us ({dispatchUs * 1000:F1} ns)");
        _output.WriteLine($"[D3D12 BENCHMARK] Round-Trip Turnaround:     {roundTripUs:F3} us ({roundTripUs * 1000:F1} ns)");
        _output.WriteLine($"[D3D12 BENCHMARK] Sample Output:             {sampleVal}");

        Assert.True(verified, $"D3D12 persistent ring buffer GPU arithmetic expected 40.0f, but got {sampleVal} (dispatch={dispatchUs:F3}us, roundTrip={roundTripUs:F3}us)");
        Assert.True(dispatchUs < 5.0, $"Dispatch latency should be under 5 us, was {dispatchUs:F3} us");
    }
}
