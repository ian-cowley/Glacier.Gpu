using Glacier.Gpu.Compilation;
using Xunit;

namespace Glacier.Gpu.Tests;

public class KernelCompilerTests
{
    [Fact]
    public void KernelCompiler_ProducesBytecode()
    {
        byte[] bytes = KernelCompiler.CompileOrPreparePtx(KernelSources.VectorAddPtx, "sm_89");
        Assert.NotNull(bytes);
        Assert.NotEmpty(bytes);
    }

    [Fact]
    public void KernelCache_CachesAndReturnsBytes()
    {
        byte[] bytes1 = KernelCache.GetOrCompile("sm_89", "TestVectorAdd", KernelSources.VectorAddPtx);
        byte[] bytes2 = KernelCache.GetOrCompile("sm_89", "TestVectorAdd", KernelSources.VectorAddPtx);

        Assert.NotNull(bytes1);
        Assert.NotNull(bytes2);
        Assert.Equal(bytes1.Length, bytes2.Length);
    }

    [Fact]
    public void GlacierComputeBridge_BuildsPtx_AndAllocatesSlab()
    {
        var builder = GlacierComputeBridge.CreatePtxBuilder("custom_gemm", 89);
        Assert.NotNull(builder);
        string ptx = builder.Build();
        Assert.Contains("custom_gemm", ptx);
        Assert.Contains("sm_89", ptx);

        using var slab = GlacierComputeBridge.CreateSlabAllocator(64);
        var ptr = slab.Allocate(256);
        Assert.False(ptr.IsNull);
        Assert.True(slab.Free(ptr));
    }
}

