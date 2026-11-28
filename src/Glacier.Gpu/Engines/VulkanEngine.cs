// <copyright file="VulkanEngine.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Compiled Vulkan compute pipeline and associated shader module / descriptor layout resources.
/// </summary>
public sealed unsafe class VulkanComputePipeline : IDisposable
{
    private readonly VulkanContext _context;
    private IntPtr _shaderModule;
    private IntPtr _descriptorSetLayout;
    private IntPtr _pipelineLayout;
    private IntPtr _pipeline;
    private bool _disposed;

    public IntPtr ShaderModuleHandle => _shaderModule;
    public IntPtr DescriptorSetLayoutHandle => _descriptorSetLayout;
    public IntPtr PipelineLayoutHandle => _pipelineLayout;
    public IntPtr PipelineHandle => _pipeline;
    public uint StorageBufferCount { get; }
    public uint PushConstantBytes { get; }
    public string EntryPoint { get; }
    public bool IsDisposed => _disposed;

    internal VulkanComputePipeline(
        VulkanContext context,
        IntPtr shaderModule,
        IntPtr descriptorSetLayout,
        IntPtr pipelineLayout,
        IntPtr pipeline,
        uint storageBufferCount,
        uint pushConstantBytes,
        string entryPoint)
    {
        _context = context;
        _shaderModule = shaderModule;
        _descriptorSetLayout = descriptorSetLayout;
        _pipelineLayout = pipelineLayout;
        _pipeline = pipeline;
        StorageBufferCount = storageBufferCount;
        PushConstantBytes = pushConstantBytes;
        EntryPoint = entryPoint;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            IntPtr device = _context.DeviceHandle;
            if (device != IntPtr.Zero)
            {
                if (!_context.IsDisposed)
                {
                    _context.Synchronize();
                }

                if (_pipeline != IntPtr.Zero)
                {
                    VulkanDriver.DestroyPipeline(device, _pipeline, IntPtr.Zero);
                    _pipeline = IntPtr.Zero;
                }

                if (_pipelineLayout != IntPtr.Zero)
                {
                    VulkanDriver.DestroyPipelineLayout(device, _pipelineLayout, IntPtr.Zero);
                    _pipelineLayout = IntPtr.Zero;
                }

                if (_descriptorSetLayout != IntPtr.Zero)
                {
                    VulkanDriver.DestroyDescriptorSetLayout(device, _descriptorSetLayout, IntPtr.Zero);
                    _descriptorSetLayout = IntPtr.Zero;
                }

                if (_shaderModule != IntPtr.Zero)
                {
                    VulkanDriver.DestroyShaderModule(device, _shaderModule, IntPtr.Zero);
                    _shaderModule = IntPtr.Zero;
                }
            }
        }
    }
}

/// <summary>
/// Universal Vulkan 1.1–1.3+ compute execution engine.
/// Zero-dependency execution across AMD Radeon, Intel Arc, and NVIDIA GPUs on Windows and Linux.
/// Automatically selects full-bandwidth <c>DEVICE_LOCAL</c> GDDR6 VRAM with a reusable 32 MB staging path
/// on discrete GPUs and zero-copy unified memory on integrated APUs.
/// </summary>
public sealed unsafe class VulkanEngine : IGpuEngine
{
    private const uint SpirvMagic = 0x07230203u;

    private VulkanContext? _context;
    private GpuDeviceInfo? _deviceInfo;
    private readonly int _deviceOrdinal;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;
    public VulkanContext Context => _context ?? throw new InvalidOperationException("Engine not initialized.");
    public bool HasCooperativeMatrix => _context?.HasCooperativeMatrix ?? false;

    /// <summary>
    /// True on discrete GPUs where pure <c>DEVICE_LOCAL</c> VRAM exceeds the host-visible BAR and staging uploads are optimal.
    /// False on unified-memory APUs where zero-copy <c>DEVICE_LOCAL | HOST_VISIBLE</c> buffers are optimal.
    /// </summary>
    public bool PreferDeviceLocalStaging => _context?.PreferDeviceLocalStaging ?? false;

    /// <summary>
    /// True when the active Vulkan device supports hardware integer dot products
    /// (Vulkan 1.3 core or <c>VK_KHR_shader_integer_dot_product</c>).
    /// </summary>
    public bool SupportsIntegerDotProduct => _context?.SupportsIntegerDotProduct ?? false;

    public VulkanEngine(int deviceOrdinal = 0)
    {
        _deviceOrdinal = deviceOrdinal;
    }

    public void Initialize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;
        _context = new VulkanContext(_deviceOrdinal);

        string devName = _context.DeviceName;
        string lower = devName.ToLowerInvariant();
        // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == 1
        bool isIntegrated = _context.PhysicalDeviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                            lower.Contains("890m") || lower.Contains("880m") ||
                            lower.Contains("780m") || lower.Contains("760m") ||
                            lower.Contains("680m") || lower.Contains("660m") ||
                            lower.Contains("apu");

        GpuDeviceType devType;
        if (_context.VendorId == 0x10DE || lower.Contains("nvidia") || lower.Contains("geforce") || lower.Contains("rtx"))
            devType = GpuDeviceType.NvidiaDiscrete;
        else if (_context.VendorId == 0x1002 || lower.Contains("amd") || lower.Contains("radeon"))
            devType = isIntegrated ? GpuDeviceType.AmdIntegratedApu : GpuDeviceType.AmdDiscrete;
        else if (_context.VendorId == 0x8086 || lower.Contains("intel") || lower.Contains("arc"))
            devType = isIntegrated ? GpuDeviceType.IntelIntegrated : GpuDeviceType.IntelDiscrete;
        else
            devType = GpuDeviceType.VulkanGeneric;

        string arch = devName;
        if (devType == GpuDeviceType.AmdIntegratedApu || devType == GpuDeviceType.AmdDiscrete)
            arch = AmdRdnaEngine.ResolveArchitecture(devName, isIntegrated);

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: devName,
            DeviceType: devType,
            Architecture: arch,
            ComputeUnitsOrSms: 0,
            TotalMemoryBytes: (long)_context.TotalVramBytes,
            IsUnifiedMemory: isIntegrated,
            SupportsPersistentRingBuffer: false,
            SupportsIntegerDotProduct: _context.SupportsIntegerDotProduct,
            PreferDeviceLocalStaging: _context.PreferDeviceLocalStaging
        );

        _initialized = true;
    }

    /// <summary>
    /// Allocates a topology-optimal compute buffer: pure <c>DEVICE_LOCAL</c> GDDR6 VRAM on discrete GPUs
    /// or cached zero-copy UMA memory on integrated APUs.
    /// </summary>
    public VulkanDeviceBuffer AllocateDeviceBuffer(ulong sizeBytes, uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Context.AllocateDeviceBufferHandle(sizeBytes, usage);
    }

    /// <summary>
    /// Allocates a topology-optimal compute buffer sized for <paramref name="elementCount"/> elements of type <typeparamref name="T"/>.
    /// </summary>
    public VulkanDeviceBuffer AllocateDeviceBuffer<T>(int elementCount, uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (elementCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be non-negative.");

        ulong bytes = (ulong)elementCount * (ulong)sizeof(T);
        return Context.AllocateDeviceBufferHandle(bytes, usage);
    }

    /// <summary>
    /// Allocates a topology-optimal compute buffer and populates it from host data using zero-copy mapping on UMA APUs
    /// or the reusable 32 MB staging buffer (<c>vkCmdCopyBuffer</c>) on discrete GPUs.
    /// </summary>
    public VulkanDeviceBuffer AllocateDeviceBuffer<T>(ReadOnlySpan<T> initialData, uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) where T : unmanaged
    {
        var buffer = AllocateDeviceBuffer<T>(initialData.Length, usage);
        try
        {
            buffer.Upload(initialData);
            return buffer;
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Compiles a SPIR-V word stream into a reusable <see cref="VulkanComputePipeline"/> bound to
    /// <paramref name="storageBufferCount"/> consecutive storage buffer descriptors (set 0, bindings <c>0..N-1</c>)
    /// and optional push constants.
    /// </summary>
    public VulkanComputePipeline CreateComputePipeline(
        ReadOnlySpan<uint> spirvWords,
        uint storageBufferCount,
        uint pushConstantBytes = 0,
        string entryPoint = "main")
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(entryPoint);
        if (spirvWords.Length < 5 || spirvWords[0] != SpirvMagic)
            throw new ArgumentException("Invalid SPIR-V module: expected >= 5 words starting with magic 0x07230203.", nameof(spirvWords));
        if (pushConstantBytes % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(pushConstantBytes), "Push constant size must be a multiple of 4 bytes.");

        VulkanContext ctx = Context;
        IntPtr device = ctx.DeviceHandle;
        IntPtr shaderModule = IntPtr.Zero;
        IntPtr descSetLayout = IntPtr.Zero;
        IntPtr pipelineLayout = IntPtr.Zero;
        IntPtr pipeline = IntPtr.Zero;

        try
        {
            fixed (uint* pCode = spirvWords)
            {
                var smInfo = new VulkanDriver.VkShaderModuleCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    codeSize = (nuint)(spirvWords.Length * sizeof(uint)),
                    pCode = (IntPtr)pCode
                };
                VulkanDriver.Check(VulkanDriver.CreateShaderModule(device, ref smInfo, IntPtr.Zero, out shaderModule), "vkCreateShaderModule");
            }

            if (storageBufferCount == 0)
            {
                var dslInfo = new VulkanDriver.VkDescriptorSetLayoutCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    bindingCount = 0,
                    pBindings = IntPtr.Zero
                };
                VulkanDriver.Check(VulkanDriver.CreateDescriptorSetLayout(device, ref dslInfo, IntPtr.Zero, out descSetLayout), "vkCreateDescriptorSetLayout");
            }
            else
            {
                VulkanDriver.VkDescriptorSetLayoutBinding[]? rentedBindings = null;
                Span<VulkanDriver.VkDescriptorSetLayoutBinding> bindings = storageBufferCount <= 32
                    ? stackalloc VulkanDriver.VkDescriptorSetLayoutBinding[(int)storageBufferCount]
                    : (rentedBindings = ArrayPool<VulkanDriver.VkDescriptorSetLayoutBinding>.Shared.Rent((int)storageBufferCount)).AsSpan(0, (int)storageBufferCount);
                try
                {
                    for (uint i = 0; i < storageBufferCount; i++)
                    {
                        bindings[(int)i] = new VulkanDriver.VkDescriptorSetLayoutBinding
                        {
                            binding = i,
                            descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount = 1,
                            stageFlags = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT
                        };
                    }

                    fixed (VulkanDriver.VkDescriptorSetLayoutBinding* pBindings = bindings)
                    {
                        var dslInfo = new VulkanDriver.VkDescriptorSetLayoutCreateInfo
                        {
                            sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                            bindingCount = storageBufferCount,
                            pBindings = (IntPtr)pBindings
                        };
                        VulkanDriver.Check(VulkanDriver.CreateDescriptorSetLayout(device, ref dslInfo, IntPtr.Zero, out descSetLayout), "vkCreateDescriptorSetLayout");
                    }
                }
                finally
                {
                    if (rentedBindings is not null)
                    {
                        ArrayPool<VulkanDriver.VkDescriptorSetLayoutBinding>.Shared.Return(rentedBindings);
                    }
                }
            }

            IntPtr dsl = descSetLayout;
            var pushRange = new VulkanDriver.VkPushConstantRange
            {
                stageFlags = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT,
                offset = 0,
                size = pushConstantBytes
            };
            var plInfo = new VulkanDriver.VkPipelineLayoutCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount = 1,
                pSetLayouts = (IntPtr)(&dsl),
                pushConstantRangeCount = pushConstantBytes > 0 ? 1u : 0u,
                pPushConstantRanges = pushConstantBytes > 0 ? (IntPtr)(&pushRange) : IntPtr.Zero
            };
            VulkanDriver.Check(VulkanDriver.CreatePipelineLayout(device, ref plInfo, IntPtr.Zero, out pipelineLayout), "vkCreatePipelineLayout");

            int utf8Len = Encoding.UTF8.GetByteCount(entryPoint);
            byte[]? rented = null;
            Span<byte> entryBytes = utf8Len < 128
                ? stackalloc byte[utf8Len + 1]
                : (rented = ArrayPool<byte>.Shared.Rent(utf8Len + 1));
            try
            {
                Encoding.UTF8.GetBytes(entryPoint, entryBytes[..utf8Len]);
                entryBytes[utf8Len] = 0;

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

                    IntPtr created = IntPtr.Zero;
                    VulkanDriver.Check(VulkanDriver.CreateComputePipelines(device, IntPtr.Zero, 1, &cpInfo, IntPtr.Zero, &created), "vkCreateComputePipelines");
                    pipeline = created;
                }
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<byte>.Shared.Return(rented);
                }
            }

            return new VulkanComputePipeline(
                ctx,
                shaderModule,
                descSetLayout,
                pipelineLayout,
                pipeline,
                storageBufferCount,
                pushConstantBytes,
                entryPoint);
        }
        catch
        {
            if (pipeline != IntPtr.Zero) VulkanDriver.DestroyPipeline(device, pipeline, IntPtr.Zero);
            if (pipelineLayout != IntPtr.Zero) VulkanDriver.DestroyPipelineLayout(device, pipelineLayout, IntPtr.Zero);
            if (descSetLayout != IntPtr.Zero) VulkanDriver.DestroyDescriptorSetLayout(device, descSetLayout, IntPtr.Zero);
            if (shaderModule != IntPtr.Zero) VulkanDriver.DestroyShaderModule(device, shaderModule, IntPtr.Zero);
            throw;
        }
    }

    /// <summary>
    /// Compiles a SPIR-V byte stream into a reusable <see cref="VulkanComputePipeline"/>.
    /// </summary>
    public VulkanComputePipeline CreateComputePipeline(
        ReadOnlySpan<byte> spirvBytes,
        uint storageBufferCount,
        uint pushConstantBytes = 0,
        string entryPoint = "main")
    {
        if (spirvBytes.Length < 20 || (spirvBytes.Length & 3) != 0)
            throw new ArgumentException("SPIR-V bytecode must be at least 20 bytes and a multiple of 4 bytes.", nameof(spirvBytes));

        return CreateComputePipeline(MemoryMarshal.Cast<byte, uint>(spirvBytes), storageBufferCount, pushConstantBytes, entryPoint);
    }

    /// <summary>
    /// Dispatches a compiled <see cref="VulkanComputePipeline"/> on the active compute queue and synchronizes via <c>VkFence</c>.
    /// </summary>
    public void DispatchCompute(
        VulkanComputePipeline pipeline,
        ReadOnlySpan<VulkanDeviceBuffer> storageBuffers,
        uint groupCountX,
        uint groupCountY = 1,
        uint groupCountZ = 1,
        ReadOnlySpan<byte> pushConstants = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(pipeline);
        ObjectDisposedException.ThrowIf(pipeline.IsDisposed, pipeline);

        if ((uint)storageBuffers.Length != pipeline.StorageBufferCount)
            throw new ArgumentException($"Expected {pipeline.StorageBufferCount} storage buffers, received {storageBuffers.Length}.", nameof(storageBuffers));
        if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0)
            throw new ArgumentOutOfRangeException(nameof(groupCountX), "Dispatch workgroup dimensions must be >= 1.");
        if ((uint)pushConstants.Length != pipeline.PushConstantBytes)
            throw new ArgumentException($"Expected {pipeline.PushConstantBytes} push constant bytes, received {pushConstants.Length}.", nameof(pushConstants));

        for (int i = 0; i < storageBuffers.Length; i++)
        {
            ArgumentNullException.ThrowIfNull(storageBuffers[i], nameof(storageBuffers));
            ObjectDisposedException.ThrowIf(storageBuffers[i].IsDisposed, storageBuffers[i]);
        }

        VulkanContext ctx = Context;
        IntPtr device = ctx.DeviceHandle;
        IntPtr descPool = IntPtr.Zero;
        IntPtr allocatedSet = IntPtr.Zero;

        try
        {
            if (pipeline.StorageBufferCount > 0)
            {
                var poolSize = new VulkanDriver.VkDescriptorPoolSize
                {
                    type = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount = pipeline.StorageBufferCount
                };
                var poolInfo = new VulkanDriver.VkDescriptorPoolCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets = 1,
                    poolSizeCount = 1,
                    pPoolSizes = (IntPtr)(&poolSize)
                };
                VulkanDriver.Check(VulkanDriver.CreateDescriptorPool(device, ref poolInfo, IntPtr.Zero, out descPool), "vkCreateDescriptorPool");

                IntPtr dsl = pipeline.DescriptorSetLayoutHandle;
                var dsAlloc = new VulkanDriver.VkDescriptorSetAllocateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool = descPool,
                    descriptorSetCount = 1,
                    pSetLayouts = (IntPtr)(&dsl)
                };
                IntPtr localSet = IntPtr.Zero;
                VulkanDriver.Check(VulkanDriver.AllocateDescriptorSets(device, ref dsAlloc, &localSet), "vkAllocateDescriptorSets");
                allocatedSet = localSet;

                int count = storageBuffers.Length;
                VulkanDriver.VkDescriptorBufferInfo[]? rentedBufInfos = null;
                VulkanDriver.VkWriteDescriptorSet[]? rentedWrites = null;
                Span<VulkanDriver.VkDescriptorBufferInfo> bufInfos = count <= 32
                    ? stackalloc VulkanDriver.VkDescriptorBufferInfo[count]
                    : (rentedBufInfos = ArrayPool<VulkanDriver.VkDescriptorBufferInfo>.Shared.Rent(count)).AsSpan(0, count);
                Span<VulkanDriver.VkWriteDescriptorSet> writes = count <= 32
                    ? stackalloc VulkanDriver.VkWriteDescriptorSet[count]
                    : (rentedWrites = ArrayPool<VulkanDriver.VkWriteDescriptorSet>.Shared.Rent(count)).AsSpan(0, count);

                try
                {
                    fixed (VulkanDriver.VkDescriptorBufferInfo* pBufInfos = bufInfos)
                    fixed (VulkanDriver.VkWriteDescriptorSet* pWrites = writes)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            pBufInfos[i] = new VulkanDriver.VkDescriptorBufferInfo
                            {
                                buffer = storageBuffers[i].BufferHandle,
                                offset = 0,
                                range = storageBuffers[i].SizeBytes
                            };
                            pWrites[i] = new VulkanDriver.VkWriteDescriptorSet
                            {
                                sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                dstSet = localSet,
                                dstBinding = (uint)i,
                                descriptorCount = 1,
                                descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                pBufferInfo = (IntPtr)(&pBufInfos[i])
                            };
                        }

                        VulkanDriver.UpdateDescriptorSets(device, (uint)count, pWrites, 0, IntPtr.Zero);
                    }
                }
                finally
                {
                    if (rentedBufInfos is not null) ArrayPool<VulkanDriver.VkDescriptorBufferInfo>.Shared.Return(rentedBufInfos);
                    if (rentedWrites is not null) ArrayPool<VulkanDriver.VkWriteDescriptorSet>.Shared.Return(rentedWrites);
                }
            }

            byte[]? pushCopy = pushConstants.IsEmpty ? null : pushConstants.ToArray();
            IntPtr boundDescSet = allocatedSet;
            ctx.RecordAndSubmitCompute(cb =>
            {
                VulkanDriver.CmdBindPipeline(cb, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.PipelineHandle);
                if (boundDescSet != IntPtr.Zero)
                {
                    IntPtr set = boundDescSet;
                    VulkanDriver.CmdBindDescriptorSets(cb, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.PipelineLayoutHandle, 0, 1, &set, 0, IntPtr.Zero);
                }

                if (pushCopy != null)
                {
                    fixed (byte* pPush = pushCopy)
                    {
                        VulkanDriver.CmdPushConstants(
                            cb,
                            pipeline.PipelineLayoutHandle,
                            VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            (uint)pushCopy.Length,
                            (IntPtr)pPush);
                    }
                }

                VulkanDriver.CmdDispatch(cb, groupCountX, groupCountY, groupCountZ);

                var barrier = new VulkanDriver.VkMemoryBarrier
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    srcAccessMask = VulkanDriver.VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask = VulkanDriver.VK_ACCESS_TRANSFER_READ_BIT | VulkanDriver.VK_ACCESS_HOST_READ_BIT
                };
                VulkanDriver.CmdPipelineBarrier(
                    cb,
                    VulkanDriver.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VulkanDriver.VK_PIPELINE_STAGE_TRANSFER_BIT | VulkanDriver.VK_PIPELINE_STAGE_HOST_BIT,
                    0,
                    1,
                    &barrier,
                    0,
                    IntPtr.Zero,
                    0,
                    IntPtr.Zero);
            }, "compute dispatch");
        }
        finally
        {
            if (descPool != IntPtr.Zero)
            {
                VulkanDriver.DestroyDescriptorPool(device, descPool, IntPtr.Zero);
            }
        }
    }

    /// <summary>
    /// Compiles and dispatches a SPIR-V word compute shader in a single <c>VkFence</c>-synchronized call.
    /// </summary>
    public void DispatchCompute(
        ReadOnlySpan<uint> spirvWords,
        ReadOnlySpan<VulkanDeviceBuffer> storageBuffers,
        uint groupCountX,
        uint groupCountY = 1,
        uint groupCountZ = 1,
        ReadOnlySpan<byte> pushConstants = default,
        string entryPoint = "main")
    {
        using var pipeline = CreateComputePipeline(spirvWords, (uint)storageBuffers.Length, (uint)pushConstants.Length, entryPoint);
        DispatchCompute(pipeline, storageBuffers, groupCountX, groupCountY, groupCountZ, pushConstants);
    }

    /// <summary>
    /// Compiles and dispatches a SPIR-V byte compute shader in a single <c>VkFence</c>-synchronized call.
    /// </summary>
    public void DispatchCompute(
        ReadOnlySpan<byte> spirvBytes,
        ReadOnlySpan<VulkanDeviceBuffer> storageBuffers,
        uint groupCountX,
        uint groupCountY = 1,
        uint groupCountZ = 1,
        ReadOnlySpan<byte> pushConstants = default,
        string entryPoint = "main")
    {
        using var pipeline = CreateComputePipeline(spirvBytes, (uint)storageBuffers.Length, (uint)pushConstants.Length, entryPoint);
        DispatchCompute(pipeline, storageBuffers, groupCountX, groupCountY, groupCountZ, pushConstants);
    }

    /// <summary>
    /// Uploads host memory into a Vulkan buffer via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void UploadViaStaging(IntPtr dstBuffer, IntPtr srcHost, ulong bytes, ulong dstOffset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.UploadViaStaging(dstBuffer, srcHost, bytes, dstOffset);
    }

    /// <summary>
    /// Uploads a typed host span into a Vulkan buffer via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void UploadViaStaging<T>(IntPtr dstBuffer, ReadOnlySpan<T> srcData, ulong dstOffset = 0) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.UploadViaStaging(dstBuffer, srcData, dstOffset);
    }

    /// <summary>
    /// Downloads data from a Vulkan buffer via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void DownloadViaStaging(IntPtr dstHost, IntPtr srcBuffer, ulong bytes, ulong srcOffset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.DownloadViaStaging(dstHost, srcBuffer, bytes, srcOffset);
    }

    /// <summary>
    /// Downloads data from a Vulkan buffer into a typed host span via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void DownloadViaStaging<T>(Span<T> dstData, IntPtr srcBuffer, ulong srcOffset = 0) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Context.DownloadViaStaging(dstData, srcBuffer, srcOffset);
    }

    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _context?.Synchronize();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _context?.Dispose();
            _context = null;
            _disposed = true;
            _initialized = false;
        }
    }
}
