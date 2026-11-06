using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Read-only Windows DXGI / DirectML adapter topology descriptor engine.
/// Note: DirectMlEngine inspects adapter capabilities only and does not submit compute kernels.
/// Use <see cref="D3D12ComputeEngine"/>, <see cref="VulkanEngine"/>, <see cref="NvidiaSassEngine"/>, or <see cref="AmdRdnaEngine"/> for GPU compute execution.
/// </summary>
[Obsolete("DirectMlEngine is an adapter topology discovery helper without compute kernel submission. Use D3D12ComputeEngine or VulkanEngine for GPU compute execution.")]
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
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;

        if (!DirectMlDriver.IsAvailable())
            throw new PlatformNotSupportedException("DirectML or DirectX 12 runtime (DirectML.dll / d3d12.dll) is not available on this system.");

        var adapters = DirectMlDriver.GetAdapters();
        if (adapters.Count == 0)
            throw new PlatformNotSupportedException("No compatible hardware DirectX 12 display adapter found on this system.");

        DirectMlDriver.DxgiAdapterInfo? selected = null;
        if (_targetAdapterIndex >= 0 && _targetAdapterIndex < adapters.Count)
        {
            selected = adapters[_targetAdapterIndex];
        }
        else
        {
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
            arch = AmdRdnaEngine.ResolveArchitecture(desc, selected.IsIntegrated);
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
            ComputeUnitsOrSms: 0,
            TotalMemoryBytes: totalMem,
            IsUnifiedMemory: selected.IsIntegrated,
            SupportsPersistentRingBuffer: false
        );

        _initialized = true;
    }

    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _initialized = false;
    }
}
