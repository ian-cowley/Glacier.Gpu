// <copyright file="AmdRdnaEngine.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;
using System.Diagnostics;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Hardware engine for AMD Radeon GPUs and APUs (RDNA 3.5, RDNA 3, RDNA 2).
/// Manages ROCm/HIP zero-copy host-mapped allocations and <c>hipMemcpy</c> DMA bandwidth diagnostics.
/// Note: <see cref="AmdRdnaEngine"/> does not launch HIP compute kernels; use <see cref="D3D12ComputeEngine"/>
/// or <see cref="VulkanEngine"/> for shader execution on AMD Radeon hardware.
/// </summary>
public sealed unsafe class AmdRdnaEngine : IGpuEngine
{
    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;

    public void Initialize()
    {
        if (!HipDriver.IsAvailable())
            throw new PlatformNotSupportedException("AMD HIP driver (amdhip64.dll) is not available on this system.");

        HipDriver.Check(HipDriver.Init(0), "hipInit");
        HipDriver.Check(HipDriver.GetDeviceCount(out int count), "hipGetDeviceCount");
        if (count == 0)
            throw new InvalidOperationException("No AMD HIP-capable devices found.");

        string devName = HipDriver.GetDeviceName(0);
        HipDriver.Check(HipDriver.SetDevice(0), "hipSetDevice");

        string lowerName = devName.ToLowerInvariant();
        bool isApu = lowerName.Contains("890m") || lowerName.Contains("880m") ||
                     lowerName.Contains("780m") || lowerName.Contains("760m") ||
                     lowerName.Contains("680m") || lowerName.Contains("660m") ||
                     lowerName.Contains("graphics") || lowerName.Contains("apu");

        string arch = ResolveArchitecture(devName, isApu);

        int cuCount = HipDriver.GetMultiprocessorCount(0);
        long totalMem = (long)HipDriver.GetTotalMemory(0);

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: devName,
            DeviceType: isApu ? GpuDeviceType.AmdIntegratedApu : GpuDeviceType.AmdDiscrete,
            Architecture: arch,
            ComputeUnitsOrSms: cuCount,
            TotalMemoryBytes: totalMem,
            IsUnifiedMemory: isApu,
            SupportsPersistentRingBuffer: false
        );

        _initialized = true;
    }

    /// <summary>
    /// Allocates host memory mapped into the AMD HIP device address space via <c>hipHostMalloc</c>.
    /// </summary>
    public UnifiedMemoryBlock<T> AllocateZeroCopy<T>(int count) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Element count must be non-negative.");

        nuint bytes = (nuint)(Math.Max(1, count) * sizeof(T));

        HipDriver.Check(HipDriver.HostMalloc(
            out IntPtr hostPtr,
            bytes,
            HipDriver.HIP_HOST_MALLOC_MAPPED | HipDriver.HIP_HOST_MALLOC_PORTABLE),
            "hipHostMalloc");

        IntPtr devPtr;
        try
        {
            HipDriver.Check(HipDriver.HostGetDevicePointer(out devPtr, hostPtr, 0), "hipHostGetDevicePointer");
        }
        catch
        {
            HipDriver.HostFree(hostPtr);
            throw;
        }

        return new UnifiedMemoryBlock<T>(hostPtr, devPtr, count, p => HipDriver.HostFree(p));
    }

    /// <summary>
    /// Benchmarks <c>hipMemcpy(..., HIP_MEMCPY_DEVICE_TO_DEVICE)</c> DMA copy bandwidth from host-mapped memory
    /// into a device allocation (does not execute compute shaders).
    /// </summary>
    public (double bandwidthGbps, double latencyMs) BenchmarkUnifiedBandwidth(int sizeMb = 256)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (sizeMb <= 0)
            throw new ArgumentOutOfRangeException(nameof(sizeMb), "Buffer size in MB must be positive.");

        nuint sizeBytes = (nuint)sizeMb * 1024 * 1024;
        IntPtr hostPtr = IntPtr.Zero;
        IntPtr devOnlyPtr = IntPtr.Zero;

        try
        {
            HipDriver.Check(HipDriver.HostMalloc(
                out hostPtr,
                sizeBytes,
                HipDriver.HIP_HOST_MALLOC_MAPPED | HipDriver.HIP_HOST_MALLOC_PORTABLE),
                "hipHostMalloc");

            HipDriver.Check(HipDriver.HostGetDevicePointer(out IntPtr devPtr, hostPtr, 0), "hipHostGetDevicePointer");
            HipDriver.Check(HipDriver.Malloc(out devOnlyPtr, sizeBytes), "hipMalloc");

            Span<float> span = new Span<float>((void*)hostPtr, (int)(sizeBytes / sizeof(float)));
            span.Fill(42.0f);

            const int iterations = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                HipDriver.Check(HipDriver.Memcpy(devOnlyPtr, devPtr, sizeBytes, HipDriver.HIP_MEMCPY_DEVICE_TO_DEVICE), "hipMemcpy");
            }
            HipDriver.Check(HipDriver.DeviceSynchronize(), "hipDeviceSynchronize");
            sw.Stop();

            double totalSecs = sw.Elapsed.TotalSeconds;
            double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
            double totalGb = (double)sizeBytes * iterations / (1024.0 * 1024.0 * 1024.0);
            double gbps = totalSecs > 0 ? totalGb / totalSecs : 0.0;

            return (gbps, avgMs);
        }
        finally
        {
            if (devOnlyPtr != IntPtr.Zero) HipDriver.Free(devOnlyPtr);
            if (hostPtr != IntPtr.Zero) HipDriver.HostFree(hostPtr);
        }
    }

    /// <summary>
    /// Unsupported legacy probe: previously wrote and read host-mapped memory purely on the CPU without executing a GPU kernel.
    /// Always throws <see cref="NotSupportedException"/>; use <see cref="AllocateZeroCopy{T}"/> with <see cref="D3D12ComputeEngine"/>
    /// or <see cref="VulkanEngine"/> for genuine GPU kernel execution.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown because no HIP compute kernel is executed.</exception>
    public (double roundTripUs, bool verified) TestZeroCopyDirectAccess()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        throw new NotSupportedException(
            "TestZeroCopyDirectAccess is unsupported because host-pointer CPU writes do not exercise GPU kernel execution. " +
            "Use AllocateZeroCopy<T>() with D3D12ComputeEngine or VulkanEngine for genuine GPU compute verification.");
    }

    public void Synchronize()
    {
        if (_initialized && !_disposed)
        {
            HipDriver.DeviceSynchronize();
        }
    }

    /// <summary>
    /// Identifies the RDNA architecture family and ISA revision (including RDNA 4.0 gfx1200, RDNA 3.5, 3.0, 2.0).
    /// </summary>
    public static string ResolveArchitecture(string devName, bool isApu)
    {
        string lower = devName.ToLowerInvariant();
        if (lower.Contains("9070") || lower.Contains("9080") || lower.Contains("9090") ||
            lower.Contains("9060") || lower.Contains("rx 9") || lower.Contains("gfx1200") || lower.Contains("gfx1201"))
            return "RDNA 4.0 (gfx1200)";
        if (lower.Contains("890m") || lower.Contains("880m") || lower.Contains("gfx1150"))
            return "RDNA 3.5 (gfx1150)";
        if (lower.Contains("780m") || lower.Contains("760m") || lower.Contains("740m") ||
            lower.Contains("7900") || lower.Contains("7800") || lower.Contains("7700") || lower.Contains("7600") || lower.Contains("gfx110"))
            return "RDNA 3.0 (gfx1103)";
        if (lower.Contains("680m") || lower.Contains("660m") ||
            lower.Contains("6900") || lower.Contains("6800") || lower.Contains("6700") || lower.Contains("6600") || lower.Contains("gfx103"))
            return "RDNA 2.0 (gfx1035)";
        return isApu ? "RDNA APU" : "RDNA dGPU";
    }

    public void Dispose()
    {
        _disposed = true;
        _initialized = false;
    }
}
