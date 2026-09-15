using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Universal Vulkan 1.3+ compute execution engine with hardware tensor (VK_KHR_cooperative_matrix) support.
/// Zero-dependency execution across AMD Radeon, Intel Arc, and NVIDIA GPUs on Windows and Linux.
/// </summary>
public sealed class VulkanEngine : IGpuEngine
{
    private VulkanContext? _context;
    private GpuDeviceInfo? _deviceInfo;
    private readonly int _deviceOrdinal;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;
    public VulkanContext Context => _context ?? throw new InvalidOperationException("Engine not initialized.");
    public bool HasCooperativeMatrix => _context?.HasCooperativeMatrix ?? false;

    public VulkanEngine(int deviceOrdinal = 0)
    {
        _deviceOrdinal = deviceOrdinal;
    }

    public void Initialize()
    {
        if (_initialized) return;
        _context = new VulkanContext(_deviceOrdinal);

        string devName = _context.DeviceName;
        string lower = devName.ToLowerInvariant();
        bool isApu = lower.Contains("890m") || lower.Contains("880m") ||
                     lower.Contains("780m") || lower.Contains("760m") ||
                     lower.Contains("680m") || lower.Contains("660m") ||
                     lower.Contains("graphics") || lower.Contains("apu");

        GpuDeviceType devType;
        if (lower.Contains("nvidia") || lower.Contains("geforce") || lower.Contains("rtx"))
            devType = GpuDeviceType.NvidiaDiscrete;
        else if (isApu)
            devType = GpuDeviceType.AmdIntegratedApu;
        else if (lower.Contains("amd") || lower.Contains("radeon"))
            devType = GpuDeviceType.AmdDiscrete;
        else if (lower.Contains("intel") || lower.Contains("arc"))
            devType = lower.Contains("arc") ? GpuDeviceType.IntelDiscrete : GpuDeviceType.IntelIntegrated;
        else
            devType = GpuDeviceType.VulkanGeneric;

        string arch = devName;
        if (lower.Contains("890m") || lower.Contains("880m"))
            arch = "RDNA 3.5 (gfx1150)";
        else if (lower.Contains("680m") || lower.Contains("660m"))
            arch = "RDNA 2.0 (gfx1035)";

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: devName,
            DeviceType: devType,
            Architecture: arch,
            ComputeUnitsOrSms: 0,
            TotalMemoryBytes: (long)_context.TotalVramBytes,
            IsUnifiedMemory: isApu,
            SupportsPersistentRingBuffer: false
        );

        _initialized = true;
    }

    public void Synchronize()
    {
        // VulkanContext manages device synchronization through fences and queues
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _context?.Dispose();
            _context = null;
            _disposed = true;
            _initialized = false;
        }
    }
}
