namespace Glacier.Gpu.Common;

/// <summary>
/// Hardware specifications and topology of an active GPU device.
/// </summary>
public record GpuDeviceInfo(
    string DeviceName,
    GpuDeviceType DeviceType,
    string Architecture,
    int ComputeUnitsOrSms,
    long TotalMemoryBytes,
    bool IsUnifiedMemory,
    bool SupportsPersistentRingBuffer
);
