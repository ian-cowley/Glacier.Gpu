using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;

namespace Glacier.Gpu.Factory;

/// <summary>
/// Discovers system GPU topology and instantiates optimal compute engines.
/// </summary>
public static class GpuEngineFactory
{
    public static bool HasNvidiaGpu => CuDriver.IsAvailable();
    public static bool HasAmdGpu => HipDriver.IsAvailable() || HasAmdD3D12 || HasVulkan;
    public static bool HasDirectMl => DirectMlDriver.IsAvailable();
    public static bool HasIntelGpu => DirectMlDriver.HasIntelGpu;
    public static bool HasD3D12 => OperatingSystem.IsWindows();
    public static bool HasVulkan => VulkanDriver.IsAvailable();

    public static bool HasAmdD3D12
    {
        get
        {
            if (!OperatingSystem.IsWindows()) return false;
            try
            {
                using var f = Vortice.DXGI.DXGI.CreateDXGIFactory1<Vortice.DXGI.IDXGIFactory4>();
                for (uint i = 0; f.EnumAdapters1(i, out Vortice.DXGI.IDXGIAdapter1 a).Success; i++)
                {
                    var desc = a.Description1;
                    bool isAmd = (desc.Flags & Vortice.DXGI.AdapterFlags.Software) == 0 &&
                                 (desc.VendorId == 0x1002 || desc.Description.Contains("Radeon", StringComparison.OrdinalIgnoreCase));
                    a.Dispose();
                    if (isAmd) return true;
                }
                return false;
            }
            catch { return false; }
        }
    }

    public static IGpuEngine CreateOptimalEngine()
    {
        if (HasNvidiaGpu && HasAmdGpu)
        {
            var het = new HeterogeneousEngine();
            het.Initialize();
            return het;
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

        if (HasAmdGpu && HipDriver.IsAvailable())
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

        if (HasDirectMl)
        {
            // Accelerated DirectML engine for Intel Arc, AMD iGPUs, or any DX12 GPU
            var dml = new DirectMlEngine();
            dml.Initialize();
            return dml;
        }

        if (HasD3D12)
        {
            var d3d = new D3D12ComputeEngine();
            d3d.Initialize();
            return d3d;
        }

        throw new PlatformNotSupportedException("No supported bare-metal GPU driver found on this system.");
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

    public static DirectMlEngine? TryCreateIntelEngine()
    {
        if (!HasIntelGpu || !HasDirectMl) return null;
        try
        {
            var adapters = DirectMlDriver.GetAdapters();
            for (int i = 0; i < adapters.Count; i++)
            {
                if (adapters[i].IsIntel)
                {
                    var dml = new DirectMlEngine(i);
                    dml.Initialize();
                    return dml;
                }
            }
            return null;
        }
        catch
        {
            return null;
        }
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