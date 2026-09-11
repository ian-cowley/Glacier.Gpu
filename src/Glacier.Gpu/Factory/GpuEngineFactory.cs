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
}
