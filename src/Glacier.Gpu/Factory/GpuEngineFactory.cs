using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;

namespace Glacier.Gpu.Factory;

/// <summary>
/// Discovers system GPU topology and instantiates real bare-metal hardware compute engines.
/// </summary>
public static class GpuEngineFactory
{
    public static bool HasNvidiaGpu => CuDriver.IsAvailable();
    public static bool HasAmdGpu => HipDriver.IsAvailable() || HasAmdD3D12 || VulkanContext.HasDeviceForVendor(0x1002);
    public static bool HasDirectMl => DirectMlDriver.IsAvailable() && DirectMlDriver.HasHardwareAdapter;
    public static bool HasIntelGpu => DirectMlDriver.HasIntelGpu || VulkanContext.HasDeviceForVendor(0x8086);
    public static bool HasD3D12 => OperatingSystem.IsWindows() && DirectMlDriver.HasHardwareAdapter;
    public static bool HasVulkan => VulkanContext.IsSupported;
    public static bool HasAmdD3D12 => OperatingSystem.IsWindows() && DirectMlDriver.HasAmdGpu;

    public static IGpuEngine CreateOptimalEngine()
    {
        if (HasNvidiaGpu && HipDriver.IsAvailable())
        {
            try
            {
                var het = new HeterogeneousEngine();
                het.Initialize();
                return het;
            }
            catch { }
        }

        if (HasNvidiaGpu)
        {
            var nv = new NvidiaSassEngine();
            nv.Initialize();
            return nv;
        }

        if (HasAmdD3D12)
        {
            var d3d = new D3D12ComputeEngine();
            d3d.Initialize();
            return d3d;
        }

        if (HipDriver.IsAvailable())
        {
            var amd = new AmdRdnaEngine();
            amd.Initialize();
            return amd;
        }

        if (HasVulkan)
        {
            try
            {
                var vk = new VulkanEngine();
                vk.Initialize();
                return vk;
            }
            catch { }
        }

        if (HasD3D12)
        {
            var d3d = new D3D12ComputeEngine();
            d3d.Initialize();
            return d3d;
        }

        throw new PlatformNotSupportedException("No supported bare-metal GPU compute driver found on this system.");
    }

    public static NvidiaSassEngine? TryCreateNvidiaEngine()
    {
        if (!HasNvidiaGpu) return null;
        try
        {
            var nv = new NvidiaSassEngine();
            nv.Initialize();
            return nv;
        }
        catch
        {
            return null;
        }
    }

    public static IGpuEngine? TryCreateAmdEngine()
    {
        if (HasAmdD3D12)
        {
            try
            {
                var d3d = new D3D12ComputeEngine();
                d3d.Initialize();
                return d3d;
            }
            catch { }
        }

        if (HipDriver.IsAvailable())
        {
            try
            {
                var amd = new AmdRdnaEngine();
                amd.Initialize();
                return amd;
            }
            catch { }
        }

        return null;
    }

    public static D3D12ComputeEngine? TryCreateD3D12Engine(int adapterIndex = -1)
    {
        if (!HasD3D12) return null;
        try
        {
            var d3d = new D3D12ComputeEngine(adapterIndex);
            d3d.Initialize();
            return d3d;
        }
        catch
        {
            return null;
        }
    }

    public static HeterogeneousEngine? TryCreateHeterogeneousEngine()
    {
        if (!HasNvidiaGpu || !HipDriver.IsAvailable()) return null;
        try
        {
            var het = new HeterogeneousEngine();
            het.Initialize();
            return het;
        }
        catch
        {
            return null;
        }
    }

    [Obsolete("DirectMlEngine is a read-only adapter topology probe without compute dispatch. Use TryCreateD3D12Engine or TryCreateVulkanEngine for GPU compute execution.")]
    public static DirectMlEngine? TryCreateDirectMlEngine(int adapterIndex = -1)
    {
        if (!HasDirectMl) return null;
        try
        {
            var dml = new DirectMlEngine(adapterIndex);
            dml.Initialize();
            return dml;
        }
        catch
        {
            return null;
        }
    }

    public static IGpuEngine? TryCreateIntelEngine()
    {
        if (!HasIntelGpu) return null;
        if (OperatingSystem.IsWindows())
        {
            try
            {
                var adapters = DirectMlDriver.GetAdapters();
                for (int i = 0; i < adapters.Count; i++)
                {
                    if (adapters[i].IsIntel)
                    {
                        var d3d = new D3D12ComputeEngine(adapters[i].Index);
                        d3d.Initialize();
                        return d3d;
                    }
                }
            }
            catch { }
        }

        return TryCreateVulkanEngine();
    }

    public static VulkanEngine? TryCreateVulkanEngine()
    {
        if (!HasVulkan) return null;
        try
        {
            var vk = new VulkanEngine();
            vk.Initialize();
            return vk;
        }
        catch
        {
            return null;
        }
    }
}
