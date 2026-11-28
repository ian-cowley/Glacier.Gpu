using System;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.Factory;
using Xunit;

namespace Glacier.Gpu.Tests;

public class DriverDetectionTests
{
    [Fact]
    public void DriverAvailability_ContractsAndErrorGuards_AreEnforced()
    {
        // Probing availability must match factory properties consistently without throwing.
        Assert.Equal(CuDriver.IsAvailable(), GpuEngineFactory.HasNvidiaGpu);
        Assert.Equal(VulkanContext.IsSupported, GpuEngineFactory.HasVulkan);
        if (!OperatingSystem.IsWindows())
        {
            Assert.False(DirectMlDriver.IsAvailable());
            Assert.False(GpuEngineFactory.HasD3D12);
        }

        // Canonical driver constants must match hardware/ABI specifications.
        Assert.Equal(63, HipDriver.HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        Assert.Equal(16, CuDriver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        Assert.Equal(75, CuDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        Assert.Equal(76, CuDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);

        // Driver error checkers must succeed on 0 and throw InvalidOperationException on non-zero status codes.
        CuDriver.Check(0, "noop");
        HipDriver.Check(0, "noop");
        VulkanDriver.Check(VulkanDriver.VK_SUCCESS, "noop");

        Assert.Throws<InvalidOperationException>(() => CuDriver.Check(1, "cuTestFail"));
        Assert.Throws<InvalidOperationException>(() => HipDriver.Check(1, "hipTestFail"));
        Assert.Throws<InvalidOperationException>(() => VulkanDriver.Check(-1, "vkTestFail"));

        // Managed buffer wrappers must validate arguments before invoking native exports.
        Assert.Throws<ArgumentNullException>(() => CuDriver.DeviceGetName((byte[])null!, 16, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => CuDriver.DeviceGetName(new byte[8], -1, 0));
        Assert.Throws<ArgumentNullException>(() => HipDriver.DeviceGetName((byte[])null!, 16, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => HipDriver.DeviceGetName(new byte[8], -1, 0));
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
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
    [Trait("Hardware", "Gpu")]
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
    [Trait("Hardware", "Gpu")]
    public void DirectMlDriver_DoesNotThrowAndDetectsAdapters()
    {
        if (!OperatingSystem.IsWindows())
        {
            Assert.False(DirectMlDriver.IsAvailable());
            return;
        }

        if (!DirectMlDriver.IsAvailable()) return;

        var adapters = DirectMlDriver.GetAdapters();
        Assert.NotEmpty(adapters);
        foreach (var a in adapters)
        {
            Assert.False(string.IsNullOrWhiteSpace(a.Description));
            Assert.True(a.VendorId > 0);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void VulkanDriver_DoesNotThrowAndDetectsDevices()
    {
        if (!VulkanDriver.IsAvailable() || !VulkanContext.IsSupported) return;

        using var engine = new VulkanEngine();
        engine.Initialize();

        Assert.True(engine.IsInitialized);
        Assert.False(string.IsNullOrWhiteSpace(engine.DeviceInfo.DeviceName));
        Assert.True(engine.DeviceInfo.TotalMemoryBytes > 0);
        Assert.NotEqual(IntPtr.Zero, engine.Context.QueueFenceHandle);
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
