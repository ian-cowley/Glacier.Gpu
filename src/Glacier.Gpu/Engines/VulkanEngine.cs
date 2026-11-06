using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Universal Vulkan 1.2+ compute execution engine.
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
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;
        _context = new VulkanContext(_deviceOrdinal);

        string devName = _context.DeviceName;
        string lower = devName.ToLowerInvariant();
        // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == 1
        bool isIntegrated = _context.PhysicalDeviceType == 1 ||
                            lower.Contains("890m") || lower.Contains("880m") ||
                            lower.Contains("780m") || lower.Contains("760m") ||
                            lower.Contains("680m") || lower.Contains("660m") ||
                            lower.Contains("apu");

        GpuDeviceType devType;
        if (_context.VendorId == 0x10DE || lower.Contains("nvidia") || lower.Contains("geforce") || lower.Contains("rtx"))
            devType = GpuDeviceType.NvidiaDiscrete;
        else if (_context.VendorId == 0x1002 || lower.Contains("amd") || lower.Contains("radeon"))
            devType = isIntegrated ? GpuDeviceType.AmdIntegratedApu : GpuDeviceType.AmdDiscrete;
        else if (_context.VendorId == 0x8086 || lower.Contains("intel") || lower.Contains("arc"))
            devType = isIntegrated ? GpuDeviceType.IntelIntegrated : GpuDeviceType.IntelDiscrete;
        else
            devType = GpuDeviceType.VulkanGeneric;

        string arch = devName;
        if (devType == GpuDeviceType.AmdIntegratedApu || devType == GpuDeviceType.AmdDiscrete)
            arch = AmdRdnaEngine.ResolveArchitecture(devName, isIntegrated);

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: devName,
            DeviceType: devType,
            Architecture: arch,
            ComputeUnitsOrSms: 0,
            TotalMemoryBytes: (long)_context.TotalVramBytes,
            IsUnifiedMemory: isIntegrated,
            SupportsPersistentRingBuffer: false
        );

        _initialized = true;
    }

    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _context?.Synchronize();
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
