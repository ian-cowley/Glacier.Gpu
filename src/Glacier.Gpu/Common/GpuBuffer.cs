// <copyright file="GpuBuffer.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Gpu.Common;

using System;
using System.Runtime.CompilerServices;
using Glacier.Gpu.Drivers;
using Vortice.Direct3D12;

/// <summary>
/// Contiguous unmanaged GPU device memory buffer (VRAM) allocated directly on hardware.
/// Eliminates host memory duplication and provides zero-copy device-resident execution.
/// </summary>
/// <typeparam name="T">Unmanaged element type (float, half, int, etc.).</typeparam>
public sealed unsafe class GpuBuffer<T> : IDisposable where T : unmanaged
{
    private IntPtr _devicePointer;
    private ulong _gpuVirtualAddress;
    private readonly int _elementCount;
    private readonly nuint _sizeInBytes;
    private readonly GpuDeviceType _deviceType;
    private readonly Action<IntPtr>? _freeAction;
    private readonly IDisposable? _underlyingResource;
    private bool _disposed;

    /// <summary>Gets the raw device pointer (CUdeviceptr for CUDA, native handle for Vulkan).</summary>
    public IntPtr DevicePointer => _devicePointer;

    /// <summary>Gets the 64-bit GPU Virtual Address (Direct3D 12 / CUDA).</summary>
    public ulong GpuVirtualAddress => _gpuVirtualAddress;

    /// <summary>Gets the total element count allocated.</summary>
    public int ElementCount => _elementCount;

    /// <summary>Gets the total buffer size in bytes.</summary>
    public nuint SizeInBytes => _sizeInBytes;

    /// <summary>Gets the hardware device type.</summary>
    public GpuDeviceType DeviceType => _deviceType;

    /// <summary>Gets the underlying native resource object, if applicable.</summary>
    public object? UnderlyingResource => _underlyingResource;

    /// <summary>Gets a value indicating whether this buffer has been disposed.</summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuBuffer{T}"/> class.
    /// </summary>
    public GpuBuffer(
        IntPtr devicePointer,
        ulong gpuVirtualAddress,
        int elementCount,
        GpuDeviceType deviceType,
        Action<IntPtr>? freeAction = null,
        IDisposable? underlyingResource = null)
    {
        _devicePointer = devicePointer;
        _gpuVirtualAddress = gpuVirtualAddress;
        _elementCount = elementCount;
        _sizeInBytes = (nuint)((ulong)elementCount * (ulong)sizeof(T));
        _deviceType = deviceType;
        _freeAction = freeAction;
        _underlyingResource = underlyingResource;
    }

    private static readonly Lock s_cudaLock = new();
    private static IntPtr s_cudaContext;

    /// <summary>
    /// Ensures that an NVIDIA CUDA context is active on the current thread.
    /// </summary>
    public static void EnsureCudaContext()
    {
        if (!CuDriver.IsAvailable()) return;

        if (CuDriver.CtxGetCurrent(out IntPtr current) == 0 && current != IntPtr.Zero)
        {
            return;
        }

        lock (s_cudaLock)
        {
            if (CuDriver.CtxGetCurrent(out current) == 0 && current != IntPtr.Zero)
            {
                return;
            }

            if (s_cudaContext != IntPtr.Zero)
            {
                CuDriver.CtxSetCurrent(s_cudaContext);
                return;
            }

            if (CuDriver.Init(0) != 0) return;
            if (CuDriver.DeviceGetCount(out int count) != 0 || count == 0) return;
            if (CuDriver.DeviceGet(out int dev, 0) != 0) return;

            if (CuDriver.CtxCreate(out s_cudaContext, 0, dev) == 0)
            {
                CuDriver.CtxSetCurrent(s_cudaContext);
            }
        }
    }

    /// <summary>
    /// Allocates device-resident VRAM on the NVIDIA CUDA driver.
    /// </summary>
    public static GpuBuffer<T> AllocateCuda(int elementCount)
    {
        if (elementCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be non-negative.");
        if (!CuDriver.IsAvailable())
            throw new PlatformNotSupportedException("NVIDIA CUDA driver is unavailable.");

        EnsureCudaContext();

        nuint bytes = (nuint)((ulong)elementCount * (ulong)sizeof(T));
        int res = CuDriver.MemAlloc(out IntPtr dptr, bytes);
        if (res != 0 || dptr == IntPtr.Zero)
            throw new OutOfMemoryException($"Failed to allocate {bytes} bytes in CUDA VRAM. Error: {res}");

        return new GpuBuffer<T>(dptr, (ulong)dptr, elementCount, GpuDeviceType.NvidiaDiscrete, ptr =>
        {
            if (CuDriver.IsAvailable())
            {
                EnsureCudaContext();
                CuDriver.MemFree(ptr);
            }
        });
    }

    /// <summary>
    /// Allocates device-resident VRAM on a Direct3D 12 device default heap (D3D12_HEAP_TYPE_DEFAULT).
    /// </summary>
    public static GpuBuffer<T> AllocateD3D12(ID3D12Device device, int elementCount, ResourceFlags flags = ResourceFlags.None)
    {
        ArgumentNullException.ThrowIfNull(device, nameof(device));
        if (elementCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be non-negative.");

        ulong bytes = (ulong)elementCount * (ulong)sizeof(T);
        var desc = ResourceDescription.Buffer(bytes, flags);
        var heapProps = new HeapProperties(HeapType.Default);
        var resource = device.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.Common);

        return new GpuBuffer<T>(
            IntPtr.Zero,
            resource.GPUVirtualAddress,
            elementCount,
            GpuDeviceType.DirectMlGeneric,
            _ => resource.Dispose(),
            resource);
    }

    /// <summary>
    /// Uploads host data into GPU device memory across PCIe.
    /// </summary>
    public void CopyFromHost(ReadOnlySpan<T> hostSpan, IntPtr stream = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (hostSpan.Length != _elementCount)
            throw new ArgumentException($"Host span length ({hostSpan.Length}) does not match buffer element count ({_elementCount}).");

        fixed (T* pHost = hostSpan)
        {
            if (_deviceType == GpuDeviceType.NvidiaDiscrete)
            {
                CopyFromHostCuda(pHost, stream);
            }
            else
            {
                throw new NotSupportedException($"Direct host copy not supported for device type {_deviceType}.");
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void CopyFromHostCuda(T* pHost, IntPtr stream)
    {
        if (!CuDriver.IsAvailable())
            throw new PlatformNotSupportedException("NVIDIA CUDA driver is unavailable.");

        EnsureCudaContext();
        if (stream == IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.MemcpyHtoD(_devicePointer, (IntPtr)pHost, _sizeInBytes), "cuMemcpyHtoD");
        }
        else
        {
            CuDriver.Check(CuDriver.MemcpyHtoDAsync(_devicePointer, (IntPtr)pHost, _sizeInBytes, stream), "cuMemcpyHtoDAsync");
        }
    }

    /// <summary>
    /// Downloads GPU device memory back to host memory across PCIe.
    /// </summary>
    public void CopyToHost(Span<T> hostSpan, IntPtr stream = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (hostSpan.Length != _elementCount)
            throw new ArgumentException($"Host span length ({hostSpan.Length}) does not match buffer element count ({_elementCount}).");

        fixed (T* pHost = hostSpan)
        {
            if (_deviceType == GpuDeviceType.NvidiaDiscrete)
            {
                CopyToHostCuda(pHost, stream);
            }
            else
            {
                throw new NotSupportedException($"Direct host download not supported for device type {_deviceType}.");
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void CopyToHostCuda(T* pHost, IntPtr stream)
    {
        if (!CuDriver.IsAvailable())
            throw new PlatformNotSupportedException("NVIDIA CUDA driver is unavailable.");

        EnsureCudaContext();
        if (stream == IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pHost, _devicePointer, _sizeInBytes), "cuMemcpyDtoH");
        }
        else
        {
            CuDriver.Check(CuDriver.MemcpyDtoHAsync((IntPtr)pHost, _devicePointer, _sizeInBytes, stream), "cuMemcpyDtoHAsync");
        }
    }

    /// <summary>
    /// Performs pure GPU-to-GPU device memory copy in resident VRAM without PCIe host transfer.
    /// </summary>
    public void CopyFromDevice(GpuBuffer<T> source, IntPtr stream = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(source, nameof(source));
        ObjectDisposedException.ThrowIf(source.IsDisposed, source);
        if (source.ElementCount != _elementCount)
            throw new ArgumentException($"Source element count ({source.ElementCount}) does not match destination element count ({_elementCount}).");

        if (_deviceType == GpuDeviceType.NvidiaDiscrete && source.DeviceType == GpuDeviceType.NvidiaDiscrete)
        {
            CopyFromDeviceCuda(source, stream);
        }
        else
        {
            throw new NotSupportedException($"GPU-to-GPU copy not supported between device types {_deviceType} and {source.DeviceType}.");
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void CopyFromDeviceCuda(GpuBuffer<T> source, IntPtr stream)
    {
        if (!CuDriver.IsAvailable())
            throw new PlatformNotSupportedException("NVIDIA CUDA driver is unavailable.");

        EnsureCudaContext();
        if (stream == IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.MemcpyDtoD(_devicePointer, source.DevicePointer, _sizeInBytes), "cuMemcpyDtoD");
        }
        else
        {
            CuDriver.Check(CuDriver.MemcpyDtoDAsync(_devicePointer, source.DevicePointer, _sizeInBytes, stream), "cuMemcpyDtoDAsync");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_devicePointer != IntPtr.Zero && _freeAction != null)
            {
                _freeAction(_devicePointer);
                _devicePointer = IntPtr.Zero;
            }
            _underlyingResource?.Dispose();
            _gpuVirtualAddress = 0;
        }
    }
}
