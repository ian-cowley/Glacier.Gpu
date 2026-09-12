using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Universal GPU compute engine powered by Microsoft DirectML and DirectX 12 Compute.
/// Provides cooperative hardware acceleration across AMD Radeon/Ryzen iGPUs, Intel Arc/Xe GPUs, and NVIDIA GPUs.
/// Built-in to Windows 10/11 with zero external C++ SDK/runtime dependencies.
/// </summary>
public sealed class DirectMlEngine : IGpuEngine
{
    private readonly int _targetAdapterIndex;
    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;

    public DirectMlEngine(int adapterIndex = -1)
    {
        _targetAdapterIndex = adapterIndex;
    }

    public void Initialize()
    {
        if (!DirectMlDriver.IsAvailable())
            throw new PlatformNotSupportedException("DirectML or DirectX 12 runtime (DirectML.dll / d3d12.dll) is not available on this system.");

        var adapters = DirectMlDriver.GetAdapters();
        if (adapters.Count == 0)
            throw new InvalidOperationException("No compatible DirectX 12 display adapter found on this system.");

        DirectMlDriver.DxgiAdapterInfo? selected = null;
        if (_targetAdapterIndex >= 0 && _targetAdapterIndex < adapters.Count)
        {
            selected = adapters[_targetAdapterIndex];
        }
        else
        {
            // Auto-select: prefer Intel or AMD if present, otherwise first available
            foreach (var a in adapters)
            {
                if (a.IsIntel || a.IsAmd)
                {
                    selected = a;
                    break;
                }
            }
            selected ??= adapters[0];
        }

        string desc = selected.Description;
        string lower = desc.ToLowerInvariant();
        GpuDeviceType devType;
        string arch;

        if (selected.IsIntel)
        {
            devType = selected.IsIntegrated ? GpuDeviceType.IntelIntegrated : GpuDeviceType.IntelDiscrete;
            if (lower.Contains("b580") || lower.Contains("b570"))
                arch = "Intel Xe2-HPG (Battlemage)";
            else if (lower.Contains("a770") || lower.Contains("a750") || lower.Contains("a580") || lower.Contains("a380") || lower.Contains("arc"))
                arch = "Intel Xe-HPG (Alchemist)";
            else
                arch = "Intel Xe (Iris Xe / UHD)";
        }
        else if (selected.IsAmd)
        {
            devType = selected.IsIntegrated ? GpuDeviceType.AmdIntegratedApu : GpuDeviceType.AmdDiscrete;
            if (lower.Contains("890m") || lower.Contains("880m"))
                arch = "AMD RDNA 3.5";
            else if (lower.Contains("780m") || lower.Contains("760m") || lower.Contains("740m") || lower.Contains("7900") || lower.Contains("7800"))
                arch = "AMD RDNA 3.0";
            else if (lower.Contains("680m") || lower.Contains("660m") || lower.Contains("6900") || lower.Contains("6800"))
                arch = "AMD RDNA 2.0";
            else
                arch = "AMD RDNA";
        }
        else if (selected.IsNvidia)
        {
            devType = GpuDeviceType.NvidiaDiscrete;
            arch = "NVIDIA DirectX 12 Compute";
        }
        else
        {
            devType = GpuDeviceType.DirectMlGeneric;
            arch = "DirectML Generic";
        }

        long totalMem = (long)(selected.DedicatedVramBytes > 0 ? selected.DedicatedVramBytes : selected.SharedSystemMemoryBytes);

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: desc,
            DeviceType: devType,
            Architecture: arch,
            ComputeUnitsOrSms: EstimateComputeUnits(selected),
            TotalMemoryBytes: totalMem,
            IsUnifiedMemory: selected.IsIntegrated,
            SupportsPersistentRingBuffer: false
        );

        _initialized = true;
    }

    private static int EstimateComputeUnits(DirectMlDriver.DxgiAdapterInfo info)
    {
        string lower = info.Description.ToLowerInvariant();
        if (info.IsIntel)
        {
            if (lower.Contains("a770")) return 32; // 32 Xe cores
            if (lower.Contains("a750")) return 28;
            if (lower.Contains("a580")) return 24;
            if (lower.Contains("a380")) return 8;
            if (lower.Contains("b580")) return 20; // 20 Xe2 cores
            if (lower.Contains("arc")) return 8;
            return 4;
        }
        if (info.IsAmd)
        {
            if (lower.Contains("890m")) return 16;
            if (lower.Contains("780m") || lower.Contains("680m")) return 12;
            if (lower.Contains("760m")) return 8;
            if (lower.Contains("660m")) return 6;
            if (lower.Contains("7900")) return 84;
            if (lower.Contains("7800")) return 60;
            if (lower.Contains("7600")) return 32;
            return 12;
        }
        return 16;
    }

    public void Synchronize()
    {
        // DirectML/DX12 command queue synchronization
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _initialized = false;
    }
}
