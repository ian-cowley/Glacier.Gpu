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
    public static bool HasAmdGpu => HipDriver.IsAvailable();
    public static bool HasDirectMl => DirectMlDriver.IsAvailable();
    public static bool HasIntelGpu => DirectMlDriver.HasIntelGpu;

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

        if (HasAmdGpu)
        {
            var amd = new AmdRdnaEngine();
            amd.Initialize();
            return amd;
        }

        if (HasDirectMl)
        {
            // Accelerated DirectML engine for Intel Arc, AMD iGPUs, or any DX12 GPU
            var dml = new DirectMlEngine();
            dml.Initialize();
            return dml;
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

    public static AmdRdnaEngine? TryCreateAmdEngine()
    {
        if (!HasAmdGpu) return null;
        try
        {
            var amd = new AmdRdnaEngine();
            amd.Initialize();
            return amd;
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
}
