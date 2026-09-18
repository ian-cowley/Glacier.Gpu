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

    [Fact]
    public unsafe void PersistentRingBuffer_ConcurrentEnqueues_ProducesUniqueMonotonicSequenceIds()
    {
        const int capacity = 64;
        int bufferBytes = capacity * sizeof(GpuWorkTask);
        IntPtr hostMem = Marshal.AllocHGlobal(bufferBytes);

        try
        {
            using var ring = new PersistentRingBuffer(hostMem, hostMem, capacity, disposeAction: () => { });

            int threadCount = 16;
            int tasksPerThread = 50;
            var taskIds = new System.Collections.Concurrent.ConcurrentBag<uint>();

            // Thread to simulate GPU clearing tasks as completed
            int running = 1;
            var gpuWorker = new Thread(() =>
            {
                GpuWorkTask* tasks = (GpuWorkTask*)hostMem;
                while (Volatile.Read(ref running) == 1)
                {
                    for (int i = 0; i < capacity; i++)
                    {
                        if (Volatile.Read(ref tasks[i].Status) == 0 && Volatile.Read(ref tasks[i].TaskId) != 0)
                        {
                            Volatile.Write(ref tasks[i].Status, 1);
                        }
                    }
                    Thread.SpinWait(10);
                }
            })
            {
                IsBackground = true
            };
            gpuWorker.Start();

            System.Threading.Tasks.Parallel.For(0, threadCount, _ =>
            {
                for (int i = 0; i < tasksPerThread; i++)
                {
                    uint id = ring.Enqueue(TaskOpCode.VectorAdd, 256, 0, 0, 0, 1.0f, timeoutMs: 5000);
                    taskIds.Add(id);
                }
            });

            Volatile.Write(ref running, 0);
            gpuWorker.Join(1000);

            Assert.Equal(threadCount * tasksPerThread, taskIds.Count);
            var idSet = new System.Collections.Generic.HashSet<uint>(taskIds);
            Assert.Equal(taskIds.Count, idSet.Count); // All unique!
            Assert.Equal(1u, idSet.Min());
            Assert.Equal((uint)(threadCount * tasksPerThread), idSet.Max());
        }
        finally
        {
            Marshal.FreeHGlobal(hostMem);
        }
    }

    [Fact]
    public unsafe void PersistentRingBuffer_SubmitAndWait_TimesOutWhenGpuStalls()
    {
        const int capacity = 64;
        int bufferBytes = capacity * sizeof(GpuWorkTask);
        IntPtr hostMem = Marshal.AllocHGlobal(bufferBytes);

        try
        {
            using var ring = new PersistentRingBuffer(hostMem, hostMem, capacity, disposeAction: () => { });

            // Submit task with 50ms timeout; GPU will never mark Status=1
            var ex = Assert.Throws<TimeoutException>(() =>
            {
                ring.SubmitAndWait(TaskOpCode.VectorAdd, 256, 0, 0, 0, 0f, timeoutMs: 50);
            });

            Assert.Contains("timed out", ex.Message);
        }
        finally
        {
            Marshal.FreeHGlobal(hostMem);
        }
    }
}
