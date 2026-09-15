namespace Glacier.Gpu.Common;

/// <summary>
/// Hardware GPU classification type.
/// </summary>
public enum GpuDeviceType
{
    Unknown = 0,
    NvidiaDiscrete = 1,
    AmdIntegratedApu = 2,
    AmdDiscrete = 3,
    IntelDiscrete = 4,
    IntelIntegrated = 5,
    DirectMlGeneric = 6,
    VulkanGeneric = 7
}
