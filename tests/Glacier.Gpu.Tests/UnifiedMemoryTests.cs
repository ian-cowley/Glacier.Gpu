using System;
using System.Runtime.InteropServices;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Xunit;

namespace Glacier.Gpu.Tests;

public class UnifiedMemoryTests
{
    [Fact]
    public void UnifiedMemoryBlock_ProvidesSpanAndRef()
    {
        int count = 1024;
        IntPtr fakePtr = Marshal.AllocHGlobal(count * sizeof(float));

        try
        {
            using var block = new UnifiedMemoryBlock<float>(fakePtr, fakePtr, count);
            Assert.Equal(count, block.Length);
            Assert.False(block.IsDisposed);

            var span = block.Span;
            span[0] = 123.45f;
            span[100] = 678.90f;

            Assert.Equal(123.45f, block.ItemRef(0));
            Assert.Equal(678.90f, block.ItemRef(100));
        }
        finally
        {
            Marshal.FreeHGlobal(fakePtr);
        }
    }

    [Fact]
    public void AmdZeroCopy_MapsHostMemoryDirectly()
    {
        if (!HipDriver.IsAvailable()) return;

        using var engine = new AmdRdnaEngine();
        engine.Initialize();

        const int N = 2048;
        using var memory = engine.AllocateZeroCopy<float>(N);

        Assert.NotEqual(IntPtr.Zero, memory.HostPointer);
        Assert.NotEqual(IntPtr.Zero, memory.DevicePointer);

        var span = memory.Span;
        for (int i = 0; i < N; i++)
        {
            span[i] = i * 2.0f;
        }

        // Direct coherency verification
        Assert.Equal(200.0f, span[100]);
        Assert.Equal(2000.0f, span[1000]);
    }
}
