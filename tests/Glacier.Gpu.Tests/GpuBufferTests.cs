using System;
using System.Runtime.InteropServices;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Xunit;

namespace Glacier.Gpu.Tests;

public class GpuBufferTests
{
    [Fact]
    public void GpuBuffer_NegativeCount_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuBuffer<float>.AllocateCuda(-1));
    }

    [Fact]
    public void GpuBuffer_CustomAllocation_PropertiesAndDisposal()
    {
        IntPtr fakePtr = Marshal.AllocHGlobal(256 * sizeof(float));
        bool freed = false;
        try
        {
            var buffer = new GpuBuffer<float>(
                fakePtr,
                (ulong)fakePtr,
                256,
                GpuDeviceType.Unknown,
                ptr => freed = true);

            Assert.Equal(256, buffer.ElementCount);
            Assert.Equal((nuint)(256 * sizeof(float)), buffer.SizeInBytes);
            Assert.Equal(fakePtr, buffer.DevicePointer);
            Assert.Equal((ulong)fakePtr, buffer.GpuVirtualAddress);
            Assert.Equal(GpuDeviceType.Unknown, buffer.DeviceType);
            Assert.False(buffer.IsDisposed);

            buffer.Dispose();
            Assert.True(buffer.IsDisposed);
            Assert.True(freed);
        }
        finally
        {
            Marshal.FreeHGlobal(fakePtr);
        }
    }

    [Fact]
    public void GpuBuffer_DisposedBuffer_ThrowsObjectDisposedException()
    {
        var buffer = new GpuBuffer<float>(IntPtr.Zero, 0, 16, GpuDeviceType.Unknown);
        buffer.Dispose();

        Assert.True(buffer.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyFromHost(new float[16]));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyToHost(new float[16]));
    }

    [Fact]
    public void GpuBuffer_HostSpanLengthMismatch_ThrowsArgumentException()
    {
        var buffer = new GpuBuffer<float>((IntPtr)0x1234, 0x1234, 32, GpuDeviceType.NvidiaDiscrete);
        Assert.Throws<ArgumentException>(() => buffer.CopyFromHost(new float[16]));
        Assert.Throws<ArgumentException>(() => buffer.CopyToHost(new float[16]));
    }

    [Fact]
    public void GpuBuffer_Cuda_AllocateUploadDownloadRoundtrip()
    {
        if (!CuDriver.IsAvailable()) return;

        const int N = 1024;
        using var buffer = GpuBuffer<float>.AllocateCuda(N);

        Assert.NotEqual(IntPtr.Zero, buffer.DevicePointer);
        Assert.Equal(N, buffer.ElementCount);
        Assert.Equal((nuint)(N * sizeof(float)), buffer.SizeInBytes);
        Assert.Equal(GpuDeviceType.NvidiaDiscrete, buffer.DeviceType);
        Assert.False(buffer.IsDisposed);

        float[] hostUpload = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostUpload[i] = (float)(i * 1.5);
        }

        buffer.CopyFromHost(hostUpload);

        float[] hostDownload = new float[N];
        buffer.CopyToHost(hostDownload);

        for (int i = 0; i < N; i++)
        {
            Assert.Equal(hostUpload[i], hostDownload[i]);
        }
    }

    [Fact]
    public void GpuBuffer_Cuda_DeviceToDeviceCopy()
    {
        if (!CuDriver.IsAvailable()) return;

        const int N = 512;
        using var srcBuffer = GpuBuffer<float>.AllocateCuda(N);
        using var dstBuffer = GpuBuffer<float>.AllocateCuda(N);

        float[] initialData = new float[N];
        for (int i = 0; i < N; i++)
        {
            initialData[i] = 42.0f + i;
        }

        srcBuffer.CopyFromHost(initialData);

        // Perform pure GPU-to-GPU copy in resident VRAM
        dstBuffer.CopyFromDevice(srcBuffer);

        float[] downloadedData = new float[N];
        dstBuffer.CopyToHost(downloadedData);

        for (int i = 0; i < N; i++)
        {
            Assert.Equal(initialData[i], downloadedData[i]);
        }
    }

    [Fact]
    public void GpuBuffer_D3D12_AllocateCommittedResource()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var engine = new D3D12ComputeEngine();
        try
        {
            engine.Initialize();
        }
        catch
        {
            return; // Skip if no D3D12 hardware adapter available
        }

        const int N = 256;
        using var buffer = GpuBuffer<float>.AllocateD3D12(engine.Device, N);

        Assert.NotEqual(0UL, buffer.GpuVirtualAddress);
        Assert.Equal(N, buffer.ElementCount);
        Assert.NotNull(buffer.UnderlyingResource);
        Assert.False(buffer.IsDisposed);

        buffer.Dispose();
        Assert.True(buffer.IsDisposed);
    }
}
