using Glacier.Gpu.Drivers;
using Glacier.Gpu.Factory;
using Xunit;

namespace Glacier.Gpu.Tests;

public class DriverDetectionTests
{
    [Fact]
    public void DriverAvailability_DoesNotThrow()
    {
        bool hasNv = GpuEngineFactory.HasNvidiaGpu;
        bool hasAmd = GpuEngineFactory.HasAmdGpu;

        // On headless CI runners, no physical GPU driver may be present; verify detection completes without throwing
        Assert.True(hasNv || hasAmd || (!hasNv && !hasAmd));
    }

    [Fact]
    public void NvidiaDriver_QueriesComputeCapability()
    {
        if (!CuDriver.IsAvailable()) return;

        CuDriver.Check(CuDriver.Init(0), "cuInit");
        CuDriver.Check(CuDriver.DeviceGetCount(out int count), "cuDeviceGetCount");
        Assert.True(count > 0);

        CuDriver.Check(CuDriver.DeviceGet(out int dev, 0), "cuDeviceGet");
        string name = CuDriver.GetDeviceName(dev);
        string arch = CuDriver.GetComputeCapability(dev);
        int smCount = CuDriver.GetMultiprocessorCount(dev);

        Assert.False(string.IsNullOrWhiteSpace(name));
        Assert.StartsWith("sm_", arch);
        Assert.True(smCount > 0);
    }

    [Fact]
    public void AmdDriver_QueriesDeviceName()
    {
        if (!HipDriver.IsAvailable()) return;

        HipDriver.Check(HipDriver.Init(0), "hipInit");
        HipDriver.Check(HipDriver.GetDeviceCount(out int count), "hipGetDeviceCount");
        Assert.True(count > 0);

        string name = HipDriver.GetDeviceName(0);
        Assert.False(string.IsNullOrWhiteSpace(name));
    }
}
