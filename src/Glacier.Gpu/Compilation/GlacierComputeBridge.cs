namespace Glacier.Gpu.Compilation;

using System;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Memory;

/// <summary>
/// First-party compute engine bridge connecting Glacier.Gpu with Glacier.Compute.
/// Provides in-memory shader assembly, PTX emission, and high-performance slab memory pooling.
/// </summary>
public static class GlacierComputeBridge
{
    /// <summary>
    /// Creates a PTX kernel builder configured for the Glacier heterogeneous GPU runtime.
    /// </summary>
    public static PtxKernelBuilder CreatePtxBuilder(string kernelName, int targetArchSm = 89)
    {
        return new PtxKernelBuilder(kernelName).SetTargetArch($"sm_{targetArchSm}");
    }

    /// <summary>
    /// Creates a high-performance device memory slab allocator.
    /// </summary>
    public static SlabAllocator CreateSlabAllocator(int slotsPerBin = 1024)
    {
        return new SlabAllocator(slotsPerBin);
    }
}
