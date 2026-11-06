using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.Factory;
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

        var (dispatchUs, roundTripUs, verified, sampleVal) = engine.MeasureRingBufferLatency(iterations: 2_000);

        _output.WriteLine($"[BENCHMARK] Enqueue Dispatch Latency: {dispatchUs:F3} us ({dispatchUs * 1000:F1} ns)");
        _output.WriteLine($"[BENCHMARK] Round-Trip Turnaround:     {roundTripUs:F3} us ({roundTripUs * 1000:F1} ns)");
        _output.WriteLine($"[BENCHMARK] Sample Output:             {sampleVal}");

        Assert.True(verified, $"Persistent ring buffer GPU arithmetic expected 40.0f, but got {sampleVal} (dispatch={dispatchUs:F3}us, roundTrip={roundTripUs:F3}us)");
        Assert.True(dispatchUs > 0.0);
        Assert.True(roundTripUs > 0.0);
    }

    [Fact]
    public unsafe void PersistentRingBuffer_CudaMultiSlotWrapAround_CompletesOver140Tasks()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();

        const int n = 256;
        nuint bytes = (nuint)(n * sizeof(float));
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_a, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_b, bytes), "cuMemAlloc");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr d_c, bytes), "cuMemAlloc");

        try
        {
            float[] a = Enumerable.Repeat(3.0f, n).ToArray();
            float[] b = Enumerable.Repeat(7.0f, n).ToArray();
            fixed (float* pA = a, pB = b)
            {
                CuDriver.Check(CuDriver.MemcpyHtoD(d_a, (IntPtr)pA, bytes), "cuMemcpyHtoD");
                CuDriver.Check(CuDriver.MemcpyHtoD(d_b, (IntPtr)pB, bytes), "cuMemcpyHtoD");
            }

            var ring = engine.GetOrCreateRingBuffer(256);
            for (int i = 0; i < 160; i++)
            {
                ring.SubmitAndWait(TaskOpCode.VectorFma, n, (ulong)d_a, (ulong)d_b, (ulong)d_c, scalar: 5.0f, timeoutMs: 5000);
            }

            ring.Dispose();

            float[] c = new float[n];
            fixed (float* pC = c)
            {
                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pC, d_c, bytes), "cuMemcpyDtoH");
            }

            Assert.Equal(22.0f, c[0], precision: 3);
            Assert.Equal(22.0f, c[n - 1], precision: 3);
        }
        finally
        {
            engine.RingBuffer?.Dispose();
            CuDriver.MemFree(d_a);
            CuDriver.MemFree(d_b);
            CuDriver.MemFree(d_c);
        }
    }

    [Theory]
    [InlineData(100_000)]
    [InlineData(1_000_000)]
    public void PersistentRingBuffer_GridStrideLoop_ComputesLargeVectorsCorrectly(int elementCount)
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();

        var (dispatchUs, roundTripUs, verified, sampleVal) = engine.MeasureRingBufferLatency(iterations: 64, elementCount: elementCount);

        _output.WriteLine($"[LARGE VECTOR N={elementCount:N0}] Round-Trip: {roundTripUs:F3} us | Sample: {sampleVal} | Verified: {verified}");

        Assert.True(verified, $"Persistent ring buffer grid-stride failed on vector size N={elementCount}.");
        Assert.Equal(40.0f, sampleVal, precision: 3);
    }

    [Fact]
    public void D3D12_CommandListDispatch_ReportsNoPersistentRingSupport_AndExecutesCorrectly()
    {
        if (!OperatingSystem.IsWindows() || !GpuEngineFactory.HasD3D12) return;

        using var engine = new D3D12ComputeEngine();
        engine.Initialize();

        Assert.False(engine.DeviceInfo.SupportsPersistentRingBuffer);
        Assert.Throws<NotSupportedException>(() => engine.GetOrCreateRingBuffer());

        var (dispatchUs, roundTripUs, verified, sampleVal) = engine.MeasureCommandListDispatchLatency(iterations: 100, elementCount: 256);

        _output.WriteLine($"[D3D12 BENCHMARK] Adapter: {engine.DeviceInfo.DeviceName} ({engine.DeviceInfo.Architecture})");
        _output.WriteLine($"[D3D12 BENCHMARK] Command-List Submit:   {dispatchUs:F3} us");
        _output.WriteLine($"[D3D12 BENCHMARK] Round-Trip Turnaround: {roundTripUs:F3} us");
        _output.WriteLine($"[D3D12 BENCHMARK] Sample Output:         {sampleVal}");

        Assert.True(verified, $"D3D12 compute shader expected 40.0f, got {sampleVal}");
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

            int threadCount = Math.Clamp(Environment.ProcessorCount, 2, 4);
            int tasksPerThread = 50;
            var taskIds = new System.Collections.Concurrent.ConcurrentBag<uint>();

            int running = 1;
            var gpuWorker = new Thread(() =>
            {
                GpuWorkTask* tasks = (GpuWorkTask*)hostMem;
                uint expectedTaskId = 1;
                int slot = 0;
                int mask = capacity - 1;
                var spinner = new SpinWait();

                while (Volatile.Read(ref running) == 1)
                {
                    if (Volatile.Read(ref tasks[slot].TaskId) == expectedTaskId)
                    {
                        Volatile.Write(ref tasks[slot].Status, 1);
                        expectedTaskId++;
                        slot = (slot + 1) & mask;
                        spinner.Reset();
                    }
                    else
                    {
                        spinner.SpinOnce();
                    }
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
                    uint id = ring.Enqueue(TaskOpCode.VectorAdd, 256, 0, 0, 0, 1.0f, timeoutMs: 10_000);
                    taskIds.Add(id);
                }
            });

            Volatile.Write(ref running, 0);
            gpuWorker.Join(2000);

            Assert.Equal(threadCount * tasksPerThread, taskIds.Count);
            var idSet = new HashSet<uint>(taskIds);
            Assert.Equal(taskIds.Count, idSet.Count);
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
