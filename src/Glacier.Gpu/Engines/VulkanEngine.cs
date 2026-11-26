using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Universal Vulkan 1.1–1.3+ compute execution engine.
/// Zero-dependency execution across AMD Radeon, Intel Arc, and NVIDIA GPUs on Windows and Linux.
/// Automatically selects full-bandwidth <c>DEVICE_LOCAL</c> GDDR6 VRAM with a reusable 32 MB staging path
/// on discrete GPUs and zero-copy unified memory on integrated APUs.
/// </summary>
public sealed unsafe class VulkanEngine : IGpuEngine
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

    /// <summary>
    /// True on discrete GPUs where pure <c>DEVICE_LOCAL</c> VRAM exceeds the host-visible BAR and staging uploads are optimal.
    /// False on unified-memory APUs where zero-copy <c>DEVICE_LOCAL | HOST_VISIBLE</c> buffers are optimal.
    /// </summary>
    public bool PreferDeviceLocalStaging => _context?.PreferDeviceLocalStaging ?? false;

    /// <summary>
    /// True when the active Vulkan device supports hardware integer dot products
    /// (Vulkan 1.3 core or <c>VK_KHR_shader_integer_dot_product</c>).
    /// </summary>
    public bool SupportsIntegerDotProduct => _context?.SupportsIntegerDotProduct ?? false;

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
        bool isIntegrated = _context.PhysicalDeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
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
            SupportsPersistentRingBuffer: false,
            SupportsIntegerDotProduct: _context.SupportsIntegerDotProduct,
            PreferDeviceLocalStaging: _context.PreferDeviceLocalStaging
        );

        _initialized = true;
    }

    /// <summary>
    /// Allocates a topology-optimal compute buffer: pure <c>DEVICE_LOCAL</c> GDDR6 VRAM on discrete GPUs
    /// or cached zero-copy UMA memory on integrated APUs.
    /// </summary>
    public VulkanDeviceBuffer AllocateDeviceBuffer(ulong sizeBytes, uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Context.AllocateDeviceBufferHandle(sizeBytes, usage);
    }

    /// <summary>
    /// Allocates a topology-optimal compute buffer sized for <paramref name="elementCount"/> elements of type <typeparamref name="T"/>.
    /// </summary>
    public VulkanDeviceBuffer AllocateDeviceBuffer<T>(int elementCount, uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (elementCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be non-negative.");

        ulong bytes = (ulong)elementCount * (ulong)sizeof(T);
        return Context.AllocateDeviceBufferHandle(bytes, usage);
    }

    /// <summary>
    /// Allocates a topology-optimal compute buffer and populates it from host data using zero-copy mapping on UMA APUs
    /// or the reusable 32 MB staging buffer (<c>vkCmdCopyBuffer</c>) on discrete GPUs.
    /// </summary>
    public VulkanDeviceBuffer AllocateDeviceBuffer<T>(ReadOnlySpan<T> initialData, uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) where T : unmanaged
    {
        var buffer = AllocateDeviceBuffer<T>(initialData.Length, usage);
        try
        {
            buffer.Upload(initialData);
            return buffer;
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Uploads host data into a Vulkan buffer via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void UploadViaStaging(IntPtr dstBuffer, IntPtr srcHost, ulong bytes, ulong dstOffset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.UploadViaStaging(dstBuffer, srcHost, bytes, dstOffset);
    }

    /// <summary>
    /// Uploads a typed host span into a Vulkan buffer via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void UploadViaStaging<T>(IntPtr dstBuffer, ReadOnlySpan<T> srcData, ulong dstOffset = 0) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.UploadViaStaging(dstBuffer, srcData, dstOffset);
    }

    /// <summary>
    /// Downloads data from a Vulkan buffer via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void DownloadViaStaging(IntPtr dstHost, IntPtr srcBuffer, ulong bytes, ulong srcOffset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.DownloadViaStaging(dstHost, srcBuffer, bytes, srcOffset);
    }

    /// <summary>
    /// Downloads data from a Vulkan buffer into a typed host span via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void DownloadViaStaging<T>(Span<T> dstData, IntPtr srcBuffer, ulong srcOffset = 0) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.DownloadViaStaging(dstData, srcBuffer, srcOffset);
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
