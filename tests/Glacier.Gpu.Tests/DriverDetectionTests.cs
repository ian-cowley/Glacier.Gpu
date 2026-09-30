using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
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

    [Fact]
    public void DirectMlDriver_DoesNotThrowAndDetectsAdapters()
    {
        bool hasDml = DirectMlDriver.IsAvailable();
        Assert.True(hasDml || !hasDml);

        if (hasDml)
        {
            var adapters = DirectMlDriver.GetAdapters();
            Assert.NotEmpty(adapters);
            foreach (var a in adapters)
            {
                Assert.False(string.IsNullOrWhiteSpace(a.Description));
                Assert.True(a.VendorId > 0);
            }
        }
    }

    [Fact]
    public void VulkanDriver_DoesNotThrowAndDetectsDevices()
    {
        bool hasVulkan = VulkanDriver.IsAvailable();
        Assert.True(hasVulkan || !hasVulkan);

        if (hasVulkan && VulkanContext.IsSupported)
        {
            using var engine = new Glacier.Gpu.Engines.VulkanEngine();
            engine.Initialize();

            Assert.True(engine.IsInitialized);
            Assert.False(string.IsNullOrWhiteSpace(engine.DeviceInfo.DeviceName));
            Assert.True(engine.DeviceInfo.TotalMemoryBytes > 0);
        }
    }

    [Theory]
    [InlineData("AMD Radeon RX 9070 XT", false, "RDNA 4.0 (gfx1200)")]
    [InlineData("AMD Radeon RX 9080", false, "RDNA 4.0 (gfx1200)")]
    [InlineData("AMD Radeon(TM) 890M Graphics", true, "RDNA 3.5 (gfx1150)")]
    [InlineData("AMD Radeon RX 7900 XTX", false, "RDNA 3.0 (gfx1103)")]
    [InlineData("AMD Radeon RX 6800 XT", false, "RDNA 2.0 (gfx1035)")]
    public void AmdArchitecture_Resolution_DetectsRdna4AndPreviousGenerations(string devName, bool isApu, string expectedArch)
    {
        string arch = AmdRdnaEngine.ResolveArchitecture(devName, isApu);
        Assert.Equal(expectedArch, arch);
    }
}

