using System;

namespace Glacier.Gpu.Common;

/// <summary>
/// Hardware compute execution engine abstraction.
/// </summary>
public interface IGpuEngine : IDisposable
{
    GpuDeviceInfo DeviceInfo { get; }
    bool IsInitialized { get; }
    void Initialize();
    void Synchronize();
}
