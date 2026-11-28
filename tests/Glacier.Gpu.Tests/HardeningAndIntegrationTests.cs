// <copyright file="HardeningAndIntegrationTests.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading.Tasks;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Gpu.Common;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.Factory;
using Glacier.Gpu.RingBuffer;
using Vortice.Direct3D12;
using Xunit;

namespace Glacier.Gpu.Tests;

public unsafe class HardeningAndIntegrationTests
{
    [StructLayout(LayoutKind.Sequential)]
    private struct ClavierDecision
    {
        public uint ActionId;
        public float Confidence;
    }

    [Fact]
    public void CuDriver_AdditionalOverloadsAndConstants_AreConsistent()
    {
        Assert.Equal(1, CuDriver.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        Assert.Equal(16, CuDriver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        Assert.Equal(75, CuDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        Assert.Equal(76, CuDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);

        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleLoadData(out _, (byte[])null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleLoadData(out _, (string)null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.ModuleGetFunction(out _, IntPtr.Zero, (string)null!));
        Assert.Throws<ArgumentNullException>(() => CuDriver.LaunchKernel(IntPtr.Zero, 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, (IntPtr[])null!));
    }

    [Fact]
    public void VulkanDriver_FenceConstantsAndStructLayout_MatchVulkanSpec()
    {
        Assert.Equal(8, VulkanDriver.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        Assert.Equal(0x00000001u, VulkanDriver.VK_FENCE_CREATE_SIGNALED_BIT);
        Assert.Equal(1, VulkanDriver.VK_NOT_READY);
        Assert.Equal(1u, VulkanDriver.VK_TRUE);
        Assert.Equal(IntPtr.Size == 8 ? 24 : 16, sizeof(VulkanDriver.VkFenceCreateInfo));
    }

    [Fact]
    public void KernelCache_FullSha256_ExtensionSeparation_AtomicWrite_AndPathValidation()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "GlacierGpuKernelCacheTests_" + Guid.NewGuid().ToString("N"));
        try
        {
            string norm89 = KernelCompiler.NormalizePtxDirectives(KernelSources.VectorAddPtx, "sm_89");
            string norm80 = KernelCompiler.NormalizePtxDirectives(KernelSources.VectorAddPtx, "sm_80");
            string key89 = KernelCache.ComputeCacheKey("sm_89", norm89);
            string key80 = KernelCache.ComputeCacheKey("sm_80", norm80);

            Assert.Equal(64, key89.Length);
            Assert.Equal(64, key80.Length);
            Assert.NotEqual(key89, key80);
            Assert.All(key89, c => Assert.True((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F')));

            // Path traversal & invalid segment rejections
            Assert.Throws<ArgumentException>(() => KernelCache.GetOrCompile("../escape", "vec_add", KernelSources.VectorAddPtx, tempRoot));
            Assert.Throws<ArgumentException>(() => KernelCache.GetOrCompile("sm_89/../etc", "vec_add", KernelSources.VectorAddPtx, tempRoot));
            Assert.Throws<ArgumentException>(() => KernelCache.GetOrCompile("sm_89", "../vec_add", KernelSources.VectorAddPtx, tempRoot));
            Assert.Throws<ArgumentException>(() => KernelCache.GetOrCompile("sm_89", "a/b", KernelSources.VectorAddPtx, tempRoot));
            Assert.Throws<ArgumentException>(() => KernelCache.GetOrCompile("sm_89", "a\\b", KernelSources.VectorAddPtx, tempRoot));
            Assert.Throws<ArgumentException>(() => KernelCache.GetOrCompile("sm_89", ".", KernelSources.VectorAddPtx, tempRoot));
            Assert.Throws<ArgumentException>(() => KernelCache.GetOrCompile("", "vec_add", KernelSources.VectorAddPtx, tempRoot));

            byte[] compiled = KernelCache.GetOrCompile("sm_89", "vector_add", KernelSources.VectorAddPtx, tempRoot);
            Assert.NotEmpty(compiled);

            string archDir = Path.Combine(tempRoot, "sm_89");
            string[] tmpFiles = Directory.GetFiles(archDir, "*.tmp");
            Assert.Empty(tmpFiles);

            bool hasPtxas = KernelCompiler.LocatePtxas() is not null;
            string expectedExt = hasPtxas ? ".cubin" : ".ptx";
            string unexpectedExt = hasPtxas ? ".ptx" : ".cubin";
            string expectedFile = Path.Combine(archDir, $"vector_add_{key89}{expectedExt}");
            string unexpectedFile = Path.Combine(archDir, $"vector_add_{key89}{unexpectedExt}");

            Assert.True(File.Exists(expectedFile));
            Assert.False(File.Exists(unexpectedFile));
            if (hasPtxas)
            {
                Assert.True(KernelCache.HasElfMagic(compiled));

                // Corrupt .cubin must not be trusted; KernelCache must recompile and overwrite atomically.
                File.WriteAllBytes(expectedFile, [0x01, 0x02, 0x03, 0x04]);
                byte[] recovered = KernelCache.GetOrCompile("sm_89", "vector_add", KernelSources.VectorAddPtx, tempRoot);
                Assert.True(KernelCache.HasElfMagic(recovered));
                Assert.Empty(Directory.GetFiles(archDir, "*.tmp"));
            }
            else
            {
                Assert.False(KernelCache.HasElfMagic(compiled));
            }
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                try { Directory.Delete(tempRoot, recursive: true); } catch { }
            }
        }
    }

    [Fact]
    public void PersistentRingBuffer_UnsupportedOpCodes_ThrowWithoutAdvancingSequence()
    {
        const int capacity = 4;
        void* mem = NativeMemory.AllocZeroed((nuint)(capacity * sizeof(GpuWorkTask)));
        try
        {
            using var ring = new PersistentRingBuffer((IntPtr)mem, (IntPtr)mem, capacity, disposeAction: () => { });

#pragma warning disable CS0618
            Assert.Throws<NotSupportedException>(() => ring.Enqueue(TaskOpCode.MatrixTile, 64, 0, 0, 0));
            Assert.Throws<NotSupportedException>(() => ring.Enqueue(TaskOpCode.ReduceSum, 64, 0, 0, 0));
            Assert.Throws<NotSupportedException>(() => ring.Enqueue(TaskOpCode.CustomKernel, 64, 0, 0, 0));
#pragma warning restore CS0618
            Assert.Throws<NotSupportedException>(() => ring.Enqueue((TaskOpCode)999, 64, 0, 0, 0));

            // Subsequent valid enqueue must still receive TaskId == 1 (sequence counter untouched).
            uint id1 = ring.Enqueue(TaskOpCode.VectorAdd, 64, 0, 0, 0);
            Assert.Equal(1u, id1);
        }
        finally
        {
            NativeMemory.Free(mem);
        }
    }

    [Fact]
    public void GlacierComputeBridge_SpirvPtxDxbcAndDxilGuard_WorkCorrectly()
    {
        Instruction[] addKernel =
        [
            new((int)ComputeOpcode.GlobalId, 0, 0, 0),
            new((int)ComputeOpcode.LoadGlobal, 0, 0, 1),
            new((int)ComputeOpcode.LoadGlobal, 1, 0, 2),
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.StoreGlobal, 3, 0, 0),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        ];

        byte[] spirvBytes = GlacierComputeBridge.CompileSpirV(addKernel);
        uint[] spirvWords = GlacierComputeBridge.CompileSpirVWords(addKernel);
        Assert.True(spirvWords.Length >= 5);
        Assert.Equal(0x07230203u, spirvWords[0]);
        Assert.Equal(spirvBytes.Length, spirvWords.Length * sizeof(uint));
        Assert.True(GlacierComputeBridge.ValidateSpirV(spirvBytes, out string? error), error);

        string ptx = GlacierComputeBridge.CompilePtx(addKernel, kernelName: "bridge_vec_add", targetArch: "sm_89");
        Assert.Contains(".visible .entry bridge_vec_add", ptx);
        Assert.Contains(".target sm_89", ptx);

        byte[] fakeBitcode = [0x42, 0x43, 0xC0, 0xDE, 0x00, 0x00, 0x00, 0x00];
        byte[] dxbc = GlacierComputeBridge.PackagePrecompiledDxbc(fakeBitcode);
        Assert.True(dxbc.Length >= 32);
        Assert.Equal(0x43425844u, BitConverter.ToUInt32(dxbc, 0)); // 'DXBC'

        var builder = GlacierComputeBridge.CreateBytecodeBuilder();
        Assert.Throws<NotSupportedException>(() => builder.BuildDxil(ShaderStage.Compute, addKernel));
    }

    [Fact]
    public void BackendHonesty_DirectMlObsoleteAndAmdZeroCopyThrow()
    {
#pragma warning disable CS0618
        var obsoleteAttr = typeof(DirectMlEngine).GetCustomAttribute<ObsoleteAttribute>();
#pragma warning restore CS0618
        Assert.NotNull(obsoleteAttr);

        using var amd = new AmdRdnaEngine();
        Assert.Throws<NotSupportedException>(() => amd.TestZeroCopyDirectAccess());
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void ArgmaxReductionPtx_EmptyK0_WritesZeroWithoutIllegalMemoryAccess()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");

        byte[] moduleBytes = KernelCache.GetOrCompile(engine.DeviceInfo.Architecture, "ArgmaxReduction_test", KernelSources.ArgmaxReductionPtx);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr mod, moduleBytes), "cuModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fn, mod, "argmax_reduction_f32"), "cuModuleGetFunction");

        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dDecision, (nuint)sizeof(ClavierDecision)), "cuMemAlloc");
        try
        {
            CuDriver.Check(CuDriver.MemsetD8(dDecision, 0xFF, (nuint)sizeof(ClavierDecision)), "cuMemsetD8");

            // Pass d_logits = IntPtr.Zero deliberately: k_classes == 0 must exit before any deref.
            IntPtr logitsPtr = IntPtr.Zero;
            IntPtr decPtr = dDecision;
            uint kClasses = 0;
            float temp = 1.0f;
            void** args = stackalloc void*[4];
            args[0] = &logitsPtr;
            args[1] = &decPtr;
            args[2] = &kClasses;
            args[3] = &temp;

            CuDriver.Check(CuDriver.LaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero), "cuLaunchKernel");
            CuDriver.Check(CuDriver.CtxSynchronize(), "cuCtxSynchronize");

            ClavierDecision decision = default;
            CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)(&decision), dDecision, (nuint)sizeof(ClavierDecision)), "cuMemcpyDtoH");

            Assert.Equal(0u, decision.ActionId);
            Assert.Equal(0.0f, decision.Confidence);
        }
        finally
        {
            CuDriver.CtxSetCurrent(engine.ContextHandle);
            CuDriver.MemFree(dDecision);
            CuDriver.ModuleUnload(mod);
        }
    }

    [Theory]
    [Trait("Hardware", "Gpu")]
    [InlineData(1, 64, 1.0f)]
    [InlineData(7, 192, 1.0f)]
    [InlineData(7, 192, 0.0f)]
    [InlineData(7, 192, -2.0f)]
    public void ArgmaxReductionPtx_SingleAndNonPowerOfTwoTieBreakAndTempGuards_MatchCpuOracle(
        int kClasses,
        uint blockDimX,
        float temperature)
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");

        byte[] moduleBytes = KernelCache.GetOrCompile(engine.DeviceInfo.Architecture, "ArgmaxReduction_test", KernelSources.ArgmaxReductionPtx);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr mod, moduleBytes), "cuModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fn, mod, "argmax_reduction_f32"), "cuModuleGetFunction");

        float[] logits = kClasses == 1
            ? [3.25f]
            : [1.0f, 5.0f, 5.0f, 2.0f, 5.0f, -3.0f, 0.0f];

        nuint logitsBytes = (nuint)(kClasses * sizeof(float));
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dLogits, logitsBytes), "cuMemAlloc(logits)");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dDecision, (nuint)sizeof(ClavierDecision)), "cuMemAlloc(decision)");

        try
        {
            fixed (float* pLogits = logits)
            {
                CuDriver.Check(CuDriver.MemcpyHtoD(dLogits, (IntPtr)pLogits, logitsBytes), "cuMemcpyHtoD");
            }

            IntPtr logitsPtr = dLogits;
            IntPtr decPtr = dDecision;
            uint k = (uint)kClasses;
            float t = temperature;
            void** args = stackalloc void*[4];
            args[0] = &logitsPtr;
            args[1] = &decPtr;
            args[2] = &k;
            args[3] = &t;

            CuDriver.Check(CuDriver.LaunchKernel(fn, 1, 1, 1, blockDimX, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero), "cuLaunchKernel");
            CuDriver.Check(CuDriver.CtxSynchronize(), "cuCtxSynchronize");

            ClavierDecision decision = default;
            CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)(&decision), dDecision, (nuint)sizeof(ClavierDecision)), "cuMemcpyDtoH");

            var (expectedId, expectedConf) = ComputeCpuArgmaxOracle(logits, temperature);
            Assert.Equal(expectedId, decision.ActionId);
            Assert.InRange(decision.Confidence, expectedConf - 1e-3f, expectedConf + 1e-3f);
        }
        finally
        {
            CuDriver.CtxSetCurrent(engine.ContextHandle);
            CuDriver.MemFree(dLogits);
            CuDriver.MemFree(dDecision);
            CuDriver.ModuleUnload(mod);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void ArgmaxReductionPtx_Large4096_ParallelSharedReduction_MatchesCpuOracle()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");

        byte[] moduleBytes = KernelCache.GetOrCompile(engine.DeviceInfo.Architecture, "ArgmaxReduction_test", KernelSources.ArgmaxReductionPtx);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr mod, moduleBytes), "cuModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fn, mod, "argmax_reduction_f32"), "cuModuleGetFunction");

        const int kClasses = 4096;
        const float temperature = 0.7f;
        float[] logits = new float[kClasses];
        for (int i = 0; i < kClasses; i++)
        {
            logits[i] = -2.0f + ((i * 17) % 100) * 0.03f;
        }
        logits[3137] = 6.25f;

        nuint logitsBytes = (nuint)(kClasses * sizeof(float));
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dLogits, logitsBytes), "cuMemAlloc(logits)");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dDecision, (nuint)sizeof(ClavierDecision)), "cuMemAlloc(decision)");

        try
        {
            fixed (float* pLogits = logits)
            {
                CuDriver.Check(CuDriver.MemcpyHtoD(dLogits, (IntPtr)pLogits, logitsBytes), "cuMemcpyHtoD");
            }

            IntPtr logitsPtr = dLogits;
            IntPtr decPtr = dDecision;
            uint k = kClasses;
            float t = temperature;
            void** args = stackalloc void*[4];
            args[0] = &logitsPtr;
            args[1] = &decPtr;
            args[2] = &k;
            args[3] = &t;

            CuDriver.Check(CuDriver.LaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero), "cuLaunchKernel");
            CuDriver.Check(CuDriver.CtxSynchronize(), "cuCtxSynchronize");

            ClavierDecision decision = default;
            CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)(&decision), dDecision, (nuint)sizeof(ClavierDecision)), "cuMemcpyDtoH");

            var (expectedId, expectedConf) = ComputeCpuArgmaxOracle(logits, temperature);
            Assert.Equal(3137u, decision.ActionId);
            Assert.Equal(expectedId, decision.ActionId);
            Assert.InRange(decision.Confidence, expectedConf - 1e-3f, expectedConf + 1e-3f);
        }
        finally
        {
            CuDriver.CtxSetCurrent(engine.ContextHandle);
            CuDriver.MemFree(dLogits);
            CuDriver.MemFree(dDecision);
            CuDriver.ModuleUnload(mod);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void PersistentWorkerPtx_MultiSmConcurrentAndSequentialStress_ProducesExactResults()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        Assert.True(engine.DeviceInfo.ComputeUnitsOrSms > 0);
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");

        const int n = 100_000;
        nuint bytes = (nuint)(n * sizeof(float));
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dA, bytes), "cuMemAlloc(A)");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dB, bytes), "cuMemAlloc(B)");
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr dC, bytes), "cuMemAlloc(C)");

        try
        {
            float[] hostA = new float[n];
            float[] hostB = new float[n];
            for (int i = 0; i < n; i++)
            {
                hostA[i] = (i % 97) * 0.25f;
                hostB[i] = (i % 31) * 0.5f + 1.0f;
            }

            fixed (float* pA = hostA, pB = hostB)
            {
                CuDriver.Check(CuDriver.MemcpyHtoD(dA, (IntPtr)pA, bytes), "cuMemcpyHtoD(A)");
                CuDriver.Check(CuDriver.MemcpyHtoD(dB, (IntPtr)pB, bytes), "cuMemcpyHtoD(B)");
            }

            PersistentRingBuffer ring = engine.GetOrCreateRingBuffer(256);

            // 1. Multi-producer concurrent Enqueue burst across ring wrap-around
            Parallel.For(0, 4, _ =>
            {
                for (int k = 0; k < 50; k++)
                {
                    ring.Enqueue(TaskOpCode.VectorAdd, n, (ulong)dA, (ulong)dB, (ulong)dC, timeoutMs: 10_000);
                }
            });

            // Drain via Nop and verify VectorAdd output across all 100,000 elements
            ring.SubmitAndWait(TaskOpCode.Nop, 0, 0, 0, 0, timeoutMs: 5000);

            float[] addOut = new float[n];
            fixed (float* pOut = addOut)
            {
                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pOut, dC, bytes), "cuMemcpyDtoH(addOut)");
            }
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(hostA[i] + hostB[i], addOut[i], precision: 4);
            }

            // 2. Sequential VectorFma wrap-around stress
            for (int k = 0; k < 100; k++)
            {
                ring.SubmitAndWait(TaskOpCode.VectorFma, n, (ulong)dA, (ulong)dB, (ulong)dC, scalar: 3.0f, timeoutMs: 5000);
            }
            ring.Dispose();

            float[] fmaOut = new float[n];
            fixed (float* pOut = fmaOut)
            {
                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pOut, dC, bytes), "cuMemcpyDtoH(fmaOut)");
            }
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(hostA[i] * 3.0f + hostB[i], fmaOut[i], precision: 4);
            }
        }
        finally
        {
            engine.RingBuffer?.Dispose();
            CuDriver.MemFree(dA);
            CuDriver.MemFree(dB);
            CuDriver.MemFree(dC);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void VulkanEngine_CreatePipelineAndDispatchCompute_WithGlacierComputeBridgeSpirv_ProducesExactResults()
    {
        if (!VulkanDriver.IsAvailable() || !VulkanContext.IsSupported) return;

        var devices = VulkanContext.EnumerateDevices();
        int ordinal = devices.FindIndex(d => d.DeviceType != VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_CPU);
        if (ordinal < 0) return;

        using var engine = new VulkanEngine(devices[ordinal].Ordinal);
        engine.Initialize();
        Assert.NotEqual(IntPtr.Zero, engine.Context.QueueFenceHandle);

        Instruction[] addKernel =
        [
            new((int)ComputeOpcode.GlobalId, 0, 0, 0),
            new((int)ComputeOpcode.LoadGlobal, 0, 0, 1),
            new((int)ComputeOpcode.LoadGlobal, 1, 0, 2),
            new((int)ComputeOpcode.Add, 1, 2, 3),
            new((int)ComputeOpcode.StoreGlobal, 3, 0, 0),
            new((int)ComputeOpcode.Return, 0, 0, 0)
        ];

        uint[] spirvWords = GlacierComputeBridge.CompileSpirVWords(addKernel, ShaderStage.Compute, localSizeX: 64);

        const int n = 2048;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = i * 1.25f;
            b[i] = (n - i) * 0.5f;
        }

        using var bufA = engine.AllocateDeviceBuffer<float>(a);
        using var bufB = engine.AllocateDeviceBuffer<float>(b);
        using var bufC = engine.AllocateDeviceBuffer<float>(n);

        byte[] pushConstants = BitConverter.GetBytes((uint)n);
        VulkanDeviceBuffer[] buffers = [bufA, bufB, bufC];

        using var pipeline = engine.CreateComputePipeline(spirvWords, storageBufferCount: 3, pushConstantBytes: (uint)pushConstants.Length);
        engine.DispatchCompute(pipeline, buffers, groupCountX: (uint)((n + 63) / 64), pushConstants: pushConstants);
        engine.Synchronize();

        float[] result = new float[n];
        bufC.Download<float>(result);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(a[i] + b[i], result[i], precision: 4);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void D3D12ComputeEngine_DispatchHlsl_ExecutesCustomShaderWithRootConstants()
    {
        if (!OperatingSystem.IsWindows() || !GpuEngineFactory.HasD3D12) return;
        RunD3D12DispatchHlslWindows();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunD3D12DispatchHlslWindows()
    {
        using var engine = new D3D12ComputeEngine();
        engine.Initialize();

        const string hlsl = @"
            cbuffer Params : register(b0)
            {
                uint count;
                float scale;
                float bias;
            };
            RWStructuredBuffer<float> bufIn : register(u0);
            RWStructuredBuffer<float> bufOut : register(u1);

            [numthreads(64, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID)
            {
                if (id.x < count)
                {
                    bufOut[id.x] = bufIn[id.x] * scale + bias;
                }
            }
        ";

        const int n = 512;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
        {
            input[i] = i * 0.5f - 10.0f;
        }

        ulong bytes = (ulong)(n * sizeof(float));
        using ID3D12Resource bufIn = engine.AllocateDeviceBuffer(bytes);
        using ID3D12Resource bufOut = engine.AllocateDeviceBuffer(bytes);

        engine.UploadBuffer<float>(bufIn, input);

        uint[] rootConsts =
        [
            (uint)n,
            BitConverter.SingleToUInt32Bits(2.5f),
            BitConverter.SingleToUInt32Bits(-3.0f)
        ];

        ID3D12Resource[] uavs = [bufIn, bufOut];
        engine.DispatchHlsl(hlsl, "main", uavs, groupCountX: (uint)((n + 63) / 64), rootConstants: rootConsts);

        float[] output = new float[n];
        engine.DownloadBuffer<float>(bufOut, output);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(input[i] * 2.5f - 3.0f, output[i], precision: 3);
        }
    }

    private static (uint actionId, float confidence) ComputeCpuArgmaxOracle(float[] logits, float temperature)
    {
        double t = temperature <= 0.0f || float.IsNaN(temperature) ? 1.0 : temperature;
        double maxVal = double.NegativeInfinity;
        uint bestId = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double scaled = logits[i] / t;
            if (scaled > maxVal)
            {
                maxVal = scaled;
                bestId = (uint)i;
            }
        }

        double sumExp = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            sumExp += Math.Exp((logits[i] / t) - maxVal);
        }

        return (bestId, (float)(1.0 / sumExp));
    }
}
