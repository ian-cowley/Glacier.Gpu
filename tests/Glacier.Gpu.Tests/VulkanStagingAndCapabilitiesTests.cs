using System;
using System.Runtime.InteropServices;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Xunit;
using Xunit.Abstractions;

namespace Glacier.Gpu.Tests;

public unsafe class VulkanStagingAndCapabilitiesTests(ITestOutputHelper output)
{
    // Minimal Vulkan 1.1 SPIR-V compute shader: bufC[gid.x] = bufA[gid.x] + bufB[gid.x] (LocalSize 64, 1, 1)
    private static readonly uint[] VectorAddSpirv =
    [
        0x07230203, 0x00010300, 0x00070000, 0x0000001B, 0x00000000, 0x00020011, 0x00000001, 0x0003000E,
        0x00000000, 0x00000001, 0x0006000F, 0x00000005, 0x00000001, 0x6E69616D, 0x00000000, 0x00000002,
        0x00060010, 0x00000001, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00040047, 0x00000002,
        0x0000000B, 0x0000001C, 0x00040047, 0x00000003, 0x00000006, 0x00000004, 0x00030047, 0x00000004,
        0x00000003, 0x00050048, 0x00000004, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000005,
        0x00000022, 0x00000000, 0x00040047, 0x00000005, 0x00000021, 0x00000000, 0x00040047, 0x00000006,
        0x00000022, 0x00000000, 0x00040047, 0x00000006, 0x00000021, 0x00000001, 0x00040047, 0x00000007,
        0x00000022, 0x00000000, 0x00040047, 0x00000007, 0x00000021, 0x00000002, 0x00020013, 0x00000008,
        0x00030021, 0x00000009, 0x00000008, 0x00040015, 0x0000000A, 0x00000020, 0x00000000, 0x00030016,
        0x0000000B, 0x00000020, 0x00040017, 0x0000000C, 0x0000000A, 0x00000003, 0x00040020, 0x0000000D,
        0x00000001, 0x0000000C, 0x0004003B, 0x0000000D, 0x00000002, 0x00000001, 0x00040020, 0x0000000E,
        0x00000001, 0x0000000A, 0x0004002B, 0x0000000A, 0x0000000F, 0x00000000, 0x0003001D, 0x00000003,
        0x0000000B, 0x0003001E, 0x00000004, 0x00000003, 0x00040020, 0x00000010, 0x00000002, 0x00000004,
        0x0004003B, 0x00000010, 0x00000005, 0x00000002, 0x0004003B, 0x00000010, 0x00000006, 0x00000002,
        0x0004003B, 0x00000010, 0x00000007, 0x00000002, 0x00040020, 0x00000011, 0x00000002, 0x0000000B,
        0x00050036, 0x00000008, 0x00000001, 0x00000000, 0x00000009, 0x000200F8, 0x00000012, 0x00050041,
        0x0000000E, 0x00000013, 0x00000002, 0x0000000F, 0x0004003D, 0x0000000A, 0x00000014, 0x00000013,
        0x00060041, 0x00000011, 0x00000015, 0x00000005, 0x0000000F, 0x00000014, 0x0004003D, 0x0000000B,
        0x00000016, 0x00000015, 0x00060041, 0x00000011, 0x00000017, 0x00000006, 0x0000000F, 0x00000014,
        0x0004003D, 0x0000000B, 0x00000018, 0x00000017, 0x00050081, 0x0000000B, 0x00000019, 0x00000016,
        0x00000018, 0x00060041, 0x00000011, 0x0000001A, 0x00000007, 0x0000000F, 0x00000014, 0x0003003E,
        0x0000001A, 0x00000019, 0x000100FD, 0x00010038
    ];

    [Fact]
    public void VulkanDriver_TransferAndDotProductStructs_MatchSpecLayout()
    {
        Assert.Equal(0x00000001u, VulkanDriver.VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Assert.Equal(0x00000002u, VulkanDriver.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        Assert.Equal(1000280000, VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES);

        Assert.Equal(24, sizeof(VulkanDriver.VkBufferCopy));
        Assert.Equal(IntPtr.Size == 8 ? 24 : 12, sizeof(VulkanDriver.VkPhysicalDeviceShaderIntegerDotProductFeatures));
        Assert.Equal(IntPtr.Size == 8 ? 240 : 228, sizeof(VulkanDriver.VkPhysicalDeviceFeatures2));

        uint v13 = VulkanDriver.MakeVersion(1, 3, 275);
        Assert.Equal(1u, VulkanDriver.VersionMajor(v13));
        Assert.Equal(3u, VulkanDriver.VersionMinor(v13));
        Assert.Equal(275u, VulkanDriver.VersionPatch(v13));
    }

    [Fact]
    public void VulkanContext_DetectsDiscreteVsUmaStagingAndIntegerDotProduct()
    {
        if (!VulkanDriver.IsAvailable() || !VulkanContext.IsSupported) return;

        var devices = VulkanContext.EnumerateDevices();
        Assert.NotEmpty(devices);

        bool sawDiscrete = false;
        bool sawIntegrated = false;

        foreach (var dev in devices)
        {
            if (dev.DeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

            using var engine = new VulkanEngine(dev.Ordinal);
            engine.Initialize();
            var ctx = engine.Context;

            Assert.Equal(dev.Name, ctx.DeviceName);
            Assert.True(ctx.ApiVersion >= VulkanDriver.MakeVersion(1, 0, 0));
            Assert.True(ctx.TotalVramBytes > 0);
            Assert.Equal(ctx.PreferDeviceLocalStaging, engine.PreferDeviceLocalStaging);
            Assert.Equal(ctx.PreferDeviceLocalStaging, engine.DeviceInfo.PreferDeviceLocalStaging);
            Assert.Equal(ctx.SupportsIntegerDotProduct, engine.SupportsIntegerDotProduct);
            Assert.Equal(ctx.SupportsIntegerDotProduct, engine.DeviceInfo.SupportsIntegerDotProduct);

            output.WriteLine(
                $"[{dev.Ordinal}] {ctx.DeviceName} | Type={ctx.PhysicalDeviceType} | API={VulkanDriver.VersionMajor(ctx.ApiVersion)}.{VulkanDriver.VersionMinor(ctx.ApiVersion)}.{VulkanDriver.VersionPatch(ctx.ApiVersion)} | " +
                $"MaxDevLocal={ctx.MaxDeviceLocalHeapBytes / (1024 * 1024)} MB | MaxHostVisDevLocal={ctx.MaxHostVisibleDeviceLocalHeapBytes / (1024 * 1024)} MB | " +
                $"PreferDeviceLocalStaging={ctx.PreferDeviceLocalStaging} | SupportsIntegerDotProduct={ctx.SupportsIntegerDotProduct} | HasCoopMatrix={ctx.HasCooperativeMatrix}");

            if (ctx.PhysicalDeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                sawDiscrete = true;
                // Discrete GPUs with smaller BAR than VRAM (e.g., RTX 4060 256 MB BAR vs 8 GB GDDR6) prefer DEVICE_LOCAL staging
                if (ctx.MaxDeviceLocalHeapBytes > ctx.MaxHostVisibleDeviceLocalHeapBytes * 2)
                {
                    Assert.True(ctx.PreferDeviceLocalStaging);
                }
                if (ctx.ApiVersion >= VulkanDriver.MakeVersion(1, 3, 0))
                {
                    Assert.True(ctx.SupportsIntegerDotProduct);
                }
            }
            else if (ctx.PhysicalDeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            {
                sawIntegrated = true;
                // Unified-memory APUs (e.g., AMD Radeon 890M Strix Point) should NOT stage when UMA heap is host-visible
                Assert.True(engine.DeviceInfo.IsUnifiedMemory);
                if (ctx.MaxHostVisibleDeviceLocalHeapBytes > 0 && ctx.MaxDeviceLocalHeapBytes <= ctx.MaxHostVisibleDeviceLocalHeapBytes * 2)
                {
                    Assert.False(ctx.PreferDeviceLocalStaging);
                }
                if (ctx.ApiVersion >= VulkanDriver.MakeVersion(1, 3, 0))
                {
                    Assert.True(ctx.SupportsIntegerDotProduct);
                }
            }
        }

        Assert.True(sawDiscrete || sawIntegrated);
    }

    [Fact]
    public void VulkanEngine_StagingUploadAndDownload_RoundtripAndMultiChunkOnDiscreteGpu()
    {
        if (!VulkanDriver.IsAvailable() || !VulkanContext.IsSupported) return;

        int ordinal = FindStagingPreferredOrDiscreteOrdinal();
        if (ordinal < 0) return;

        using var engine = new VulkanEngine(ordinal);
        engine.Initialize();
        var ctx = engine.Context;
        output.WriteLine($"Testing staging path on [{ordinal}] {ctx.DeviceName} (PreferDeviceLocalStaging={ctx.PreferDeviceLocalStaging})");

        // 1. Small buffer staging upload + download + offset copy
        const int smallCount = 4096;
        float[] src = new float[smallCount];
        for (int i = 0; i < smallCount; i++)
        {
            src[i] = (i * 1.25f) - 512.5f;
        }

        using var devBuf = engine.AllocateDeviceBuffer<float>(smallCount);
        Assert.NotEqual(IntPtr.Zero, devBuf.BufferHandle);
        Assert.True(devBuf.IsDeviceLocal);
        if (ctx.PreferDeviceLocalStaging)
        {
            Assert.False(devBuf.IsHostVisible);
        }

        engine.UploadViaStaging<float>(devBuf.BufferHandle, src);
        ulong stagingSizeAfterFirst = ctx.StagingBufferSizeBytes;
        Assert.True(stagingSizeAfterFirst >= VulkanContext.DefaultMinStagingBytes);

        // Overwrite a middle slice with dstOffset via staging
        float[] patch = [999.5f, -123.25f, 42.75f, 3.14159f];
        const int patchIndex = 1000;
        engine.UploadViaStaging<float>(devBuf.BufferHandle, patch, dstOffset: (ulong)(patchIndex * sizeof(float)));
        // Reusable staging buffer should not grow for smaller upload
        Assert.Equal(stagingSizeAfterFirst, ctx.StagingBufferSizeBytes);

        float[] dst = new float[smallCount];
        engine.DownloadViaStaging<float>(dst, devBuf.BufferHandle);

        for (int i = 0; i < smallCount; i++)
        {
            float expected = (i >= patchIndex && i < patchIndex + patch.Length) ? patch[i - patchIndex] : src[i];
            Assert.Equal(expected, dst[i]);
        }

        // 2. Multi-chunk transfer (> 32 MB staging cap: 36 MB = 9M floats)
        const int largeCount = 9 * 1024 * 1024;
        float[] largeSrc = new float[largeCount];
        largeSrc[0] = 1.5f;
        largeSrc[4 * 1024 * 1024] = -77.25f;
        largeSrc[8 * 1024 * 1024] = 888.125f;
        largeSrc[largeCount - 1] = 12345.5f;

        using var largeBuf = engine.AllocateDeviceBuffer<float>(largeSrc);
        Assert.True(ctx.StagingBufferSizeBytes <= VulkanContext.DefaultMaxStagingBytes);
        if (ctx.PreferDeviceLocalStaging)
        {
            Assert.Equal(VulkanContext.DefaultMaxStagingBytes, ctx.StagingBufferSizeBytes);
        }

        float[] largeDst = new float[largeCount];
        largeBuf.Download<float>(largeDst);

        Assert.Equal(largeSrc[0], largeDst[0]);
        Assert.Equal(largeSrc[4 * 1024 * 1024], largeDst[4 * 1024 * 1024]);
        Assert.Equal(largeSrc[8 * 1024 * 1024], largeDst[8 * 1024 * 1024]);
        Assert.Equal(largeSrc[largeCount - 1], largeDst[largeCount - 1]);

        ctx.ReleaseStagingBuffer();
        Assert.Equal(0UL, ctx.StagingBufferSizeBytes);
    }

    [Fact]
    public void VulkanEngine_ComputeDispatch_OnDeviceLocalStagedBuffers_ProducesExactResults()
    {
        if (!VulkanDriver.IsAvailable() || !VulkanContext.IsSupported) return;

        var devices = VulkanContext.EnumerateDevices();
        int testedDevices = 0;

        foreach (var dev in devices)
        {
            if (dev.DeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

            using var engine = new VulkanEngine(dev.Ordinal);
            engine.Initialize();
            var ctx = engine.Context;

            const int N = 4096;
            float[] a = new float[N];
            float[] b = new float[N];
            for (int i = 0; i < N; i++)
            {
                a[i] = i * 0.5f;
                b[i] = (N - i) * 0.25f + 10.0f;
            }

            using var bufA = engine.AllocateDeviceBuffer<float>(a);
            using var bufB = engine.AllocateDeviceBuffer<float>(b);
            using var bufC = engine.AllocateDeviceBuffer<float>(N);

            DispatchVectorAdd(ctx, bufA, bufB, bufC, N);

            float[] result = new float[N];
            bufC.Download<float>(result);

            for (int i = 0; i < N; i++)
            {
                Assert.Equal(a[i] + b[i], result[i], precision: 4);
            }

            testedDevices++;
        }

        Assert.True(testedDevices > 0);
    }

    private static int FindStagingPreferredOrDiscreteOrdinal()
    {
        var devices = VulkanContext.EnumerateDevices();
        int discreteFallback = -1;
        int anyFallback = -1;

        foreach (var d in devices)
        {
            if (d.DeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
            if (anyFallback < 0) anyFallback = d.Ordinal;
            if (d.DeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && discreteFallback < 0)
                discreteFallback = d.Ordinal;

            using var probe = new VulkanContext(d.Ordinal);
            if (probe.PreferDeviceLocalStaging)
                return d.Ordinal;
        }

        return discreteFallback >= 0 ? discreteFallback : anyFallback;
    }

    private static void DispatchVectorAdd(
        VulkanContext ctx,
        VulkanDeviceBuffer bufA,
        VulkanDeviceBuffer bufB,
        VulkanDeviceBuffer bufC,
        int elementCount)
    {
        IntPtr device = ctx.DeviceHandle;
        IntPtr shaderModule = IntPtr.Zero;
        IntPtr descSetLayout = IntPtr.Zero;
        IntPtr pipelineLayout = IntPtr.Zero;
        IntPtr pipeline = IntPtr.Zero;
        IntPtr descPool = IntPtr.Zero;

        try
        {
            // 1. Shader module
            fixed (uint* pCode = VectorAddSpirv)
            {
                var smInfo = new VulkanDriver.VkShaderModuleCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    codeSize = (nuint)(VectorAddSpirv.Length * sizeof(uint)),
                    pCode = (IntPtr)pCode
                };
                VulkanDriver.Check(VulkanDriver.CreateShaderModule(device, ref smInfo, IntPtr.Zero, out shaderModule), "vkCreateShaderModule");
            }

            // 2. Descriptor set layout (3 storage buffers)
            var bindings = stackalloc VulkanDriver.VkDescriptorSetLayoutBinding[3];
            for (uint i = 0; i < 3; i++)
            {
                bindings[i] = new VulkanDriver.VkDescriptorSetLayoutBinding
                {
                    binding = i,
                    descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount = 1,
                    stageFlags = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT
                };
            }
            var dslInfo = new VulkanDriver.VkDescriptorSetLayoutCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount = 3,
                pBindings = (IntPtr)bindings
            };
            VulkanDriver.Check(VulkanDriver.CreateDescriptorSetLayout(device, ref dslInfo, IntPtr.Zero, out descSetLayout), "vkCreateDescriptorSetLayout");

            // 3. Pipeline layout
            IntPtr dsl = descSetLayout;
            var plInfo = new VulkanDriver.VkPipelineLayoutCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount = 1,
                pSetLayouts = (IntPtr)(&dsl)
            };
            VulkanDriver.Check(VulkanDriver.CreatePipelineLayout(device, ref plInfo, IntPtr.Zero, out pipelineLayout), "vkCreatePipelineLayout");

            // 4. Compute pipeline
            byte[] entryBytes = "main\0"u8.ToArray();
            fixed (byte* pEntry = entryBytes)
            {
                var cpInfo = new VulkanDriver.VkComputePipelineCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    stage = new VulkanDriver.VkPipelineShaderStageCreateInfo
                    {
                        sType = VulkanDriver.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        stage = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT,
                        module = shaderModule,
                        pName = (IntPtr)pEntry
                    },
                    layout = pipelineLayout
                };
                IntPtr pip = IntPtr.Zero;
                VulkanDriver.Check(VulkanDriver.CreateComputePipelines(device, IntPtr.Zero, 1, &cpInfo, IntPtr.Zero, &pip), "vkCreateComputePipelines");
                pipeline = pip;
            }

            // 5. Descriptor pool & set
            var poolSize = new VulkanDriver.VkDescriptorPoolSize
            {
                type = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount = 3
            };
            var poolInfo = new VulkanDriver.VkDescriptorPoolCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets = 1,
                poolSizeCount = 1,
                pPoolSizes = (IntPtr)(&poolSize)
            };
            VulkanDriver.Check(VulkanDriver.CreateDescriptorPool(device, ref poolInfo, IntPtr.Zero, out descPool), "vkCreateDescriptorPool");

            var dsAlloc = new VulkanDriver.VkDescriptorSetAllocateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool = descPool,
                descriptorSetCount = 1,
                pSetLayouts = (IntPtr)(&dsl)
            };
            IntPtr descSet = IntPtr.Zero;
            VulkanDriver.Check(VulkanDriver.AllocateDescriptorSets(device, ref dsAlloc, &descSet), "vkAllocateDescriptorSets");

            var bufInfos = stackalloc VulkanDriver.VkDescriptorBufferInfo[3];
            bufInfos[0] = new() { buffer = bufA.BufferHandle, offset = 0, range = bufA.SizeBytes };
            bufInfos[1] = new() { buffer = bufB.BufferHandle, offset = 0, range = bufB.SizeBytes };
            bufInfos[2] = new() { buffer = bufC.BufferHandle, offset = 0, range = bufC.SizeBytes };

            var writes = stackalloc VulkanDriver.VkWriteDescriptorSet[3];
            for (uint i = 0; i < 3; i++)
            {
                writes[i] = new VulkanDriver.VkWriteDescriptorSet
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet = descSet,
                    dstBinding = i,
                    descriptorCount = 1,
                    descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo = (IntPtr)(&bufInfos[i])
                };
            }
            VulkanDriver.UpdateDescriptorSets(device, 3, writes, 0, IntPtr.Zero);

            // 6. Record & submit compute dispatch
            IntPtr cb = ctx.CommandBufferHandle;
            VulkanDriver.Check(VulkanDriver.ResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
            var begin = new VulkanDriver.VkCommandBufferBeginInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            };
            VulkanDriver.Check(VulkanDriver.BeginCommandBuffer(cb, ref begin), "vkBeginCommandBuffer");
            VulkanDriver.CmdBindPipeline(cb, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VulkanDriver.CmdBindDescriptorSets(cb, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, IntPtr.Zero);
            VulkanDriver.CmdDispatch(cb, (uint)((elementCount + 63) / 64), 1, 1);
            VulkanDriver.Check(VulkanDriver.EndCommandBuffer(cb), "vkEndCommandBuffer");

            var submit = new VulkanDriver.VkSubmitInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount = 1,
                pCommandBuffers = (IntPtr)(&cb)
            };
            VulkanDriver.Check(VulkanDriver.QueueSubmit(ctx.QueueHandle, 1, ref submit, IntPtr.Zero), "vkQueueSubmit");
            ctx.Synchronize();
        }
        finally
        {
            ctx.Synchronize();
            if (descPool != IntPtr.Zero) VulkanDriver.DestroyDescriptorPool(device, descPool, IntPtr.Zero);
            if (pipeline != IntPtr.Zero) VulkanDriver.DestroyPipeline(device, pipeline, IntPtr.Zero);
            if (pipelineLayout != IntPtr.Zero) VulkanDriver.DestroyPipelineLayout(device, pipelineLayout, IntPtr.Zero);
            if (descSetLayout != IntPtr.Zero) VulkanDriver.DestroyDescriptorSetLayout(device, descSetLayout, IntPtr.Zero);
            if (shaderModule != IntPtr.Zero) VulkanDriver.DestroyShaderModule(device, shaderModule, IntPtr.Zero);
        }
    }
}
