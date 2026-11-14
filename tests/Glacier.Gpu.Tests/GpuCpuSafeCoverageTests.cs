using System;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Gpu.Common;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Factory;
using Glacier.Gpu.RingBuffer;
using Xunit;

namespace Glacier.Gpu.Tests;

public class GpuCpuSafeCoverageTests
{
    [Fact]
    public void GpuWorkTask_Is64ByteCacheLineAligned()
    {
        Assert.Equal(64, Marshal.SizeOf<GpuWorkTask>());
    }

    [Fact]
    public void GpuDeviceInfo_RecordEqualityAndProperties_WorkCorrectly()
    {
        var d1 = new GpuDeviceInfo("Mock GPU", GpuDeviceType.NvidiaDiscrete, "sm_89", 48, 8L << 30, false, true);
        var d2 = d1 with { };
        var d3 = d1 with { ComputeUnitsOrSms = 64 };

        Assert.Equal(d1, d2);
        Assert.NotEqual(d1, d3);
        Assert.Equal("Mock GPU", d1.DeviceName);
        Assert.True(d1.SupportsPersistentRingBuffer);
    }

    [Fact]
    public void KernelCache_GetOrCompile_CachesNormalizedPtxBytes()
    {
        byte[] first = KernelCache.GetOrCompile("sm_89", "vector_add_test", KernelSources.VectorAddPtx);
        byte[] second = KernelCache.GetOrCompile("sm_89", "vector_add_test", KernelSources.VectorAddPtx);

        Assert.NotEmpty(first);
        Assert.Equal(first, second);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-8)]
    [InlineData(3)]
    public void PersistentRingBuffer_InvalidCapacity_ThrowsArgumentException(int capacity)
    {
        Assert.Throws<ArgumentException>(() => new PersistentRingBuffer(IntPtr.Zero, IntPtr.Zero, capacity));
    }

    [Fact]
    public unsafe void PersistentRingBuffer_Dispose_InvokesCallbacksAndMarksDisposed()
    {
        void* mem = NativeMemory.AllocZeroed((nuint)(4 * Marshal.SizeOf<GpuWorkTask>()));
        bool shutdownCalled = false;
        bool disposeCalled = false;

        try
        {
            var ring = new PersistentRingBuffer(
                (IntPtr)mem,
                (IntPtr)mem,
                capacity: 4,
                shutdownAction: () => shutdownCalled = true,
                disposeAction: () => disposeCalled = true);

            Assert.Equal(4, ring.Capacity);
            Assert.False(ring.IsDisposed);

            ring.Dispose();

            Assert.True(ring.IsDisposed);
            Assert.True(shutdownCalled);
            Assert.True(disposeCalled);
        }
        finally
        {
            NativeMemory.Free(mem);
        }
    }

    [Fact]
    public void GpuEngineFactory_TryCreateMethods_DoNotThrowOnCpuOnlyOrGpuHosts()
    {
        using var nv = GpuEngineFactory.TryCreateNvidiaEngine();
        using var amd = GpuEngineFactory.TryCreateAmdEngine();
        using var d3d = GpuEngineFactory.TryCreateD3D12Engine();

        if (nv != null) Assert.NotNull(nv.DeviceInfo);
        if (amd != null) Assert.NotNull(amd.DeviceInfo);
        if (d3d != null) Assert.NotNull(d3d.DeviceInfo);
    }
}
