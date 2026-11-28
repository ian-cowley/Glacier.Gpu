namespace Glacier.Gpu.Engines;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Gpu.Drivers;

/// <summary>
/// Represents a Vulkan buffer allocation owned by a <see cref="VulkanContext"/>.
/// Automatically selects pure <c>DEVICE_LOCAL</c> GDDR6 VRAM on discrete GPUs (with reusable 32 MB staging)
/// or cached zero-copy <c>DEVICE_LOCAL | HOST_VISIBLE</c> memory on unified-memory APUs.
/// </summary>
public sealed unsafe class VulkanDeviceBuffer : IDisposable
{
    private readonly VulkanContext _owner;
    private IntPtr _buffer;
    private IntPtr _memory;
    private bool _disposed;

    public IntPtr BufferHandle => _buffer;
    public IntPtr MemoryHandle => _memory;
    public ulong SizeBytes { get; }
    public bool IsHostVisible { get; }
    public bool IsDeviceLocal { get; }
    public bool IsDisposed => _disposed;

    internal VulkanDeviceBuffer(
        VulkanContext owner,
        IntPtr buffer,
        IntPtr memory,
        ulong sizeBytes,
        bool isHostVisible,
        bool isDeviceLocal)
    {
        _owner = owner;
        _buffer = buffer;
        _memory = memory;
        SizeBytes = sizeBytes;
        IsHostVisible = isHostVisible;
        IsDeviceLocal = isDeviceLocal;
    }

    /// <summary>
    /// Uploads host data into this buffer using direct zero-copy mapping on UMA APUs
    /// or the reusable 32 MB staging buffer (<c>vkCmdCopyBuffer</c>) on discrete GPUs.
    /// </summary>
    public void Upload<T>(ReadOnlySpan<T> srcData, ulong dstOffset = 0) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ulong bytes = (ulong)srcData.Length * (ulong)sizeof(T);
        if (bytes == 0) return;
        if (dstOffset + bytes > SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(srcData), $"Upload range [{dstOffset}..{dstOffset + bytes}) exceeds buffer size ({SizeBytes} bytes).");

        fixed (T* pSrc = srcData)
        {
            _owner.UploadDeviceBuffer(_buffer, _memory, IsHostVisible, (IntPtr)pSrc, bytes, dstOffset);
        }
    }

    /// <summary>
    /// Downloads data from this buffer into host memory using direct zero-copy mapping on UMA APUs
    /// or the reusable 32 MB staging buffer (<c>vkCmdCopyBuffer</c>) on discrete GPUs.
    /// </summary>
    public void Download<T>(Span<T> dstData, ulong srcOffset = 0) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ulong bytes = (ulong)dstData.Length * (ulong)sizeof(T);
        if (bytes == 0) return;
        if (srcOffset + bytes > SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(dstData), $"Download range [{srcOffset}..{srcOffset + bytes}) exceeds buffer size ({SizeBytes} bytes).");

        fixed (T* pDst = dstData)
        {
            _owner.DownloadDeviceBuffer((IntPtr)pDst, _buffer, _memory, IsHostVisible, bytes, srcOffset);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_buffer != IntPtr.Zero || _memory != IntPtr.Zero)
            {
                _owner.DestroyBuffer(_buffer, _memory);
                _buffer = IntPtr.Zero;
                _memory = IntPtr.Zero;
            }
        }
    }
}

/// <summary>
/// Universal Vulkan 1.1–1.3+ compute context.
/// Manages instance creation (Vulkan 1.3 with safe 1.1 fallback), physical device selection,
/// discrete vs. UMA staging detection (<see cref="PreferDeviceLocalStaging"/>),
/// <c>VK_KHR_shader_integer_dot_product</c> capability detection (<see cref="SupportsIntegerDotProduct"/>),
/// reusable 32 MB staging transfers (<c>vkCmdCopyBuffer</c>), and compute queue / command pool management.
/// </summary>
public sealed unsafe class VulkanContext : IDisposable
{
    /// <summary>Maximum chunk size (32 MB) for the reusable staging buffer.</summary>
    public const ulong DefaultMaxStagingBytes = 32UL * 1024UL * 1024UL;

    /// <summary>Minimum allocation size (4 MB) when lazily creating the reusable staging buffer.</summary>
    public const ulong DefaultMinStagingBytes = 4UL * 1024UL * 1024UL;

    private readonly Lock _stagingLock = new();
    private readonly Lock _queueLock = new();
    private IntPtr _instance;
    private IntPtr _physicalDevice;
    private IntPtr _device;
    private IntPtr _computeQueue;
    private uint _computeQueueIndex;
    private IntPtr _commandPool;
    private IntPtr _commandBuffer;
    private IntPtr _queueFence;
    private bool _fenceInFlight;
    private IntPtr _stagingBuffer;
    private IntPtr _stagingMemory;
    private ulong _stagingSize;
    private VulkanDriver.VkPhysicalDeviceMemoryProperties _memProps;
    private bool _disposed;

    public string DeviceName { get; }
    public uint VendorId { get; }
    public uint PhysicalDeviceType { get; }
    public uint ApiVersion { get; }
    public uint InstanceApiVersion { get; }
    public ulong TotalVramBytes { get; }
    public ulong MaxDeviceLocalHeapBytes { get; }
    public ulong MaxHostVisibleDeviceLocalHeapBytes { get; }
    public int DeviceOrdinal { get; }
    public bool HasCooperativeMatrix { get; }

    /// <summary>
    /// True on discrete GPUs where pure <c>DEVICE_LOCAL</c> VRAM (e.g., 8–24 GB GDDR6) is much larger than
    /// the host-visible <c>DEVICE_LOCAL</c> BAR (typically 256 MB).
    /// False on unified-memory APUs (e.g., AMD Radeon 890M / Strix Point) where zero-copy UMA is optimal.
    /// </summary>
    public bool PreferDeviceLocalStaging { get; }

    /// <summary>
    /// True when the physical and logical Vulkan device supports hardware 8-bit/16-bit integer dot products
    /// via Vulkan 1.3 core or <c>VK_KHR_shader_integer_dot_product</c>.
    /// </summary>
    public bool SupportsIntegerDotProduct { get; }

    /// <summary>Current allocated size (in bytes) of the reusable staging buffer, or 0 if not yet allocated.</summary>
    public ulong StagingBufferSizeBytes => _stagingSize;

    public IntPtr DeviceHandle => _device;
    public IntPtr InstanceHandle => _instance;
    public IntPtr PhysicalDeviceHandle => _physicalDevice;
    public IntPtr QueueHandle => _computeQueue;
    public IntPtr CommandPoolHandle => _commandPool;
    public IntPtr CommandBufferHandle => _commandBuffer;
    public IntPtr QueueFenceHandle => _queueFence;
    public uint ComputeQueueIndex => _computeQueueIndex;
    internal bool IsDisposed => _disposed;

    private static bool TryCreateInstanceWithFallback(string appName, out IntPtr instance, out uint apiVersion)
    {
        Span<uint> candidateVersions =
        [
            VulkanDriver.MakeVersion(1, 3, 0),
            VulkanDriver.MakeVersion(1, 2, 0),
            VulkanDriver.MakeVersion(1, 1, 0),
            VulkanDriver.MakeVersion(1, 0, 0)
        ];

        IntPtr pAppName = Marshal.StringToHGlobalAnsi(appName);
        IntPtr pEngineName = Marshal.StringToHGlobalAnsi("GlacierEngine");
        try
        {
            foreach (uint candidate in candidateVersions)
            {
                var appInfo = new VulkanDriver.VkApplicationInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                    pApplicationName = pAppName,
                    applicationVersion = VulkanDriver.MakeVersion(1, 1, 0),
                    pEngineName = pEngineName,
                    engineVersion = VulkanDriver.MakeVersion(1, 1, 0),
                    apiVersion = candidate
                };

                var createInfo = new VulkanDriver.VkInstanceCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                    pApplicationInfo = new IntPtr(&appInfo)
                };

                int res = VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out IntPtr inst);
                if (res == VulkanDriver.VK_SUCCESS && inst != IntPtr.Zero)
                {
                    instance = inst;
                    apiVersion = candidate;
                    return true;
                }
            }
        }
        finally
        {
            Marshal.FreeHGlobal(pAppName);
            Marshal.FreeHGlobal(pEngineName);
        }

        instance = IntPtr.Zero;
        apiVersion = 0;
        return false;
    }

    public static bool IsSupported
    {
        get
        {
            try
            {
                if (!VulkanDriver.IsAvailable()) return false;
                if (!TryCreateInstanceWithFallback("Glacier.Gpu.Probe", out IntPtr inst, out _))
                    return false;

                try
                {
                    uint devCount = 0;
                    int enumRes = VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, (IntPtr*)null);
                    return enumRes == VulkanDriver.VK_SUCCESS && devCount > 0;
                }
                finally
                {
                    VulkanDriver.DestroyInstance(inst, IntPtr.Zero);
                }
            }
            catch
            {
                return false;
            }
        }
    }

    /// <summary>
    /// Enumerates all Vulkan physical devices exposed by the system driver.
    /// </summary>
    public static List<(int Ordinal, string Name, uint VendorId, uint DeviceType, ulong TotalVramBytes)> EnumerateDevices()
    {
        var result = new List<(int Ordinal, string Name, uint VendorId, uint DeviceType, ulong TotalVramBytes)>();
        try
        {
            if (!VulkanDriver.IsAvailable()) return result;
            if (!TryCreateInstanceWithFallback("Glacier.Gpu.Enum", out IntPtr inst, out _))
                return result;

            try
            {
                uint devCount = 0;
                if (VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, (IntPtr*)null) != VulkanDriver.VK_SUCCESS || devCount == 0)
                    return result;

                var devices = new IntPtr[devCount];
                if (VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, devices) != VulkanDriver.VK_SUCCESS)
                    return result;

                for (uint i = 0; i < devCount; i++)
                {
                    VulkanDriver.GetPhysicalDeviceProperties(devices[i], out var props);
                    VulkanDriver.GetPhysicalDeviceMemoryProperties(devices[i], out var memProps);
                    ulong deviceLocalMem = 0;
                    ulong allHeapsMem = 0;
                    var heaps = (VulkanDriver.VkMemoryHeap*)memProps.memoryHeaps;
                    for (uint h = 0; h < memProps.memoryHeapCount; h++)
                    {
                        allHeapsMem += heaps[h].size;
                        if ((heaps[h].flags & VulkanDriver.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                            deviceLocalMem += heaps[h].size;
                    }
                    result.Add(((int)i, props.GetDeviceName(), props.vendorID, props.deviceType, deviceLocalMem > 0 ? deviceLocalMem : allHeapsMem));
                }
            }
            finally
            {
                VulkanDriver.DestroyInstance(inst, IntPtr.Zero);
            }
        }
        catch
        {
            // Defensive: return collected devices or empty list when ICD loader fails.
        }
        return result;
    }

    public static bool HasDeviceForVendor(uint vendorId)
    {
        try
        {
            if (!VulkanDriver.IsAvailable()) return false;
            if (!TryCreateInstanceWithFallback("Glacier.Gpu.VendorProbe", out IntPtr inst, out _))
                return false;

            try
            {
                uint devCount = 0;
                if (VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, (IntPtr*)null) != VulkanDriver.VK_SUCCESS || devCount == 0)
                    return false;

                var devices = new IntPtr[devCount];
                if (VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, devices) != VulkanDriver.VK_SUCCESS)
                    return false;

                for (uint i = 0; i < devCount; i++)
                {
                    VulkanDriver.GetPhysicalDeviceProperties(devices[i], out var props);
                    // Exclude VK_PHYSICAL_DEVICE_TYPE_CPU (4)
                    if (props.vendorID == vendorId && props.deviceType != VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_CPU)
                        return true;
                }
                return false;
            }
            finally
            {
                VulkanDriver.DestroyInstance(inst, IntPtr.Zero);
            }
        }
        catch
        {
            return false;
        }
    }

    public VulkanContext(int deviceOrdinal = 0)
    {
        if (!VulkanDriver.IsAvailable())
            throw new PlatformNotSupportedException("Vulkan driver (vulkan-1.dll / libvulkan.so.1) is not installed.");

        // 1. Create Vulkan Instance (prefers 1.3, falls back safely to 1.2 / 1.1 / 1.0)
        if (!TryCreateInstanceWithFallback("Glacier.Gpu", out _instance, out uint instApiVersion))
            throw new InvalidOperationException("Failed to create Vulkan instance across API versions 1.3..1.0.");

        InstanceApiVersion = instApiVersion;

        try
        {
            // 2. Enumerate Physical Devices
            uint devCount = 0;
            VulkanDriver.EnumeratePhysicalDevices(_instance, ref devCount, (IntPtr*)null);
            if (devCount == 0)
                throw new InvalidOperationException("No Vulkan physical devices found on this system.");

            var devices = new IntPtr[devCount];
            VulkanDriver.EnumeratePhysicalDevices(_instance, ref devCount, devices);

            DeviceOrdinal = Math.Clamp(deviceOrdinal, 0, (int)devCount - 1);
            _physicalDevice = devices[DeviceOrdinal];

            VulkanDriver.GetPhysicalDeviceProperties(_physicalDevice, out var props);
            DeviceName = props.GetDeviceName();
            VendorId = props.vendorID;
            PhysicalDeviceType = props.deviceType;
            ApiVersion = props.apiVersion;

            VulkanDriver.GetPhysicalDeviceMemoryProperties(_physicalDevice, out _memProps);
            ulong deviceLocalMem = 0;
            ulong allHeapsMem = 0;
            ulong maxDeviceLocalHeap = 0;
            ulong maxDeviceLocalHostVisibleHeap = 0;
            fixed (byte* pHeaps = _memProps.memoryHeaps)
            fixed (byte* pTypes = _memProps.memoryTypes)
            {
                var heaps = (VulkanDriver.VkMemoryHeap*)pHeaps;
                var types = (VulkanDriver.VkMemoryType*)pTypes;
                for (uint h = 0; h < _memProps.memoryHeapCount; h++)
                {
                    allHeapsMem += heaps[h].size;
                    if ((heaps[h].flags & VulkanDriver.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                    {
                        deviceLocalMem += heaps[h].size;
                    }
                }

                for (uint t = 0; t < _memProps.memoryTypeCount; t++)
                {
                    uint flags = types[t].propertyFlags;
                    uint hi = types[t].heapIndex;
                    if (hi >= _memProps.memoryHeapCount) continue;
                    ulong hSize = heaps[hi].size;
                    if ((flags & VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
                    {
                        if (hSize > maxDeviceLocalHeap) maxDeviceLocalHeap = hSize;
                        if ((flags & VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && hSize > maxDeviceLocalHostVisibleHeap)
                            maxDeviceLocalHostVisibleHeap = hSize;
                    }
                }
            }

            TotalVramBytes = deviceLocalMem > 0 ? deviceLocalMem : allHeapsMem;
            MaxDeviceLocalHeapBytes = maxDeviceLocalHeap;
            MaxHostVisibleDeviceLocalHeapBytes = maxDeviceLocalHostVisibleHeap;
            PreferDeviceLocalStaging = maxDeviceLocalHeap > 0 &&
                                       (maxDeviceLocalHostVisibleHeap == 0 || (maxDeviceLocalHeap / 2) > maxDeviceLocalHostVisibleHeap);

            // 2b. Probe Vulkan 1.3 / VK_KHR_shader_integer_dot_product and VK_KHR_cooperative_matrix safely
            bool isVulkan13OrHigher = ApiVersion >= VulkanDriver.MakeVersion(1, 3, 0);
            bool hasDotProductExt = false;
            bool hasCoopMatrixExt = false;
            ProbeDeviceExtensions(_physicalDevice, out hasDotProductExt, out hasCoopMatrixExt);

            bool dotProductSupported = false;
            bool coopMatrixSupported = false;
            if (InstanceApiVersion >= VulkanDriver.MakeVersion(1, 1, 0) &&
                ApiVersion >= VulkanDriver.MakeVersion(1, 1, 0))
            {
                if (isVulkan13OrHigher || hasDotProductExt)
                {
                    dotProductSupported = QueryIntegerDotProductSupport(_physicalDevice);
                }
                if (hasCoopMatrixExt)
                {
                    coopMatrixSupported = QueryCooperativeMatrixSupport(_physicalDevice);
                }
            }

            HasCooperativeMatrix = coopMatrixSupported;

            // 3. Find Compute Queue Family
            uint qfCount = 0;
            VulkanDriver.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref qfCount, (VulkanDriver.VkQueueFamilyProperties*)null);
            var qfProps = new VulkanDriver.VkQueueFamilyProperties[qfCount];
            VulkanDriver.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref qfCount, qfProps);

            int computeIndex = -1;
            for (uint i = 0; i < qfCount; i++)
            {
                if ((qfProps[i].queueFlags & VulkanDriver.VK_QUEUE_COMPUTE_BIT) != 0)
                {
                    computeIndex = (int)i;
                    break;
                }
            }

            if (computeIndex < 0)
                throw new InvalidOperationException($"No compute queue family found on {DeviceName}");

            _computeQueueIndex = (uint)computeIndex;

            // 4. Create Logical Device with Compute Queue (enabling shaderIntegerDotProduct when supported, with safe fallback)
            float priority = 1.0f;
            var queueInfo = new VulkanDriver.VkDeviceQueueCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex = _computeQueueIndex,
                queueCount = 1,
                pQueuePriorities = new IntPtr(&priority)
            };

            var devInfo = new VulkanDriver.VkDeviceCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount = 1,
                pQueueCreateInfos = new IntPtr(&queueInfo)
            };

            int devRes = -1;
            if (dotProductSupported)
            {
                var enableDotFeatures = new VulkanDriver.VkPhysicalDeviceShaderIntegerDotProductFeatures
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
                    pNext = IntPtr.Zero,
                    shaderIntegerDotProduct = 1
                };
                devInfo.pNext = new IntPtr(&enableDotFeatures);

                if (!isVulkan13OrHigher && hasDotProductExt)
                {
                    IntPtr extNamePtr = Marshal.StringToHGlobalAnsi(VulkanDriver.VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
                    try
                    {
                        IntPtr* ppExts = stackalloc IntPtr[1];
                        ppExts[0] = extNamePtr;
                        devInfo.enabledExtensionCount = 1;
                        devInfo.ppEnabledExtensionNames = new IntPtr(ppExts);
                        devRes = VulkanDriver.CreateDevice(_physicalDevice, ref devInfo, IntPtr.Zero, out _device);
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(extNamePtr);
                    }
                }
                else
                {
                    devRes = VulkanDriver.CreateDevice(_physicalDevice, ref devInfo, IntPtr.Zero, out _device);
                }

                if (devRes != VulkanDriver.VK_SUCCESS || _device == IntPtr.Zero)
                {
                    // Safe fallback: retry logical device creation without integer dot product feature
                    dotProductSupported = false;
                    devInfo.pNext = IntPtr.Zero;
                    devInfo.enabledExtensionCount = 0;
                    devInfo.ppEnabledExtensionNames = IntPtr.Zero;
                    devRes = VulkanDriver.CreateDevice(_physicalDevice, ref devInfo, IntPtr.Zero, out _device);
                }
            }
            else
            {
                devRes = VulkanDriver.CreateDevice(_physicalDevice, ref devInfo, IntPtr.Zero, out _device);
            }

            SupportsIntegerDotProduct = dotProductSupported;

            if (devRes != VulkanDriver.VK_SUCCESS || _device == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create Vulkan logical device on {DeviceName}: code {devRes}");

            VulkanDriver.GetDeviceQueue(_device, _computeQueueIndex, 0, out _computeQueue);

            // 5. Create Compute Command Pool
            var poolInfo = new VulkanDriver.VkCommandPoolCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex = _computeQueueIndex,
                flags = 2 // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
            };

            int poolRes = VulkanDriver.CreateCommandPool(_device, ref poolInfo, IntPtr.Zero, out _commandPool);
            if (poolRes != VulkanDriver.VK_SUCCESS)
                throw new InvalidOperationException($"Failed to create Vulkan command pool: code {poolRes}");

            // 6. Allocate Primary Command Buffer
            var allocInfo = new VulkanDriver.VkCommandBufferAllocateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool = _commandPool,
                level = 0, // VK_COMMAND_BUFFER_LEVEL_PRIMARY
                commandBufferCount = 1
            };

            var cmdBufs = new IntPtr[1];
            int cmdRes = VulkanDriver.AllocateCommandBuffers(_device, ref allocInfo, cmdBufs);
            if (cmdRes != VulkanDriver.VK_SUCCESS)
                throw new InvalidOperationException($"Failed to allocate Vulkan command buffer: code {cmdRes}");

            _commandBuffer = cmdBufs[0];

            // 7. Create Reusable Queue Synchronization Fence
            var fenceInfo = new VulkanDriver.VkFenceCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                flags = 0
            };
            int fenceRes = VulkanDriver.CreateFence(_device, ref fenceInfo, IntPtr.Zero, out _queueFence);
            if (fenceRes != VulkanDriver.VK_SUCCESS || _queueFence == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create Vulkan queue fence: code {fenceRes}");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private static void ProbeDeviceExtensions(IntPtr physicalDevice, out bool hasDotProductExt, out bool hasCoopMatrixExt)
    {
        hasDotProductExt = false;
        hasCoopMatrixExt = false;
        try
        {
            uint extCount = 0;
            if (VulkanDriver.EnumerateDeviceExtensionProperties(physicalDevice, IntPtr.Zero, ref extCount, (VulkanDriver.VkExtensionProperties*)null) != VulkanDriver.VK_SUCCESS || extCount == 0)
                return;

            var exts = new VulkanDriver.VkExtensionProperties[extCount];
            if (VulkanDriver.EnumerateDeviceExtensionProperties(physicalDevice, IntPtr.Zero, ref extCount, exts) != VulkanDriver.VK_SUCCESS)
                return;

            for (uint i = 0; i < extCount; i++)
            {
                string extName = exts[i].GetExtensionName();
                if (string.Equals(extName, VulkanDriver.VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, StringComparison.Ordinal))
                {
                    hasDotProductExt = true;
                }
                else if (string.Equals(extName, VulkanDriver.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, StringComparison.Ordinal))
                {
                    hasCoopMatrixExt = true;
                }
            }
        }
        catch
        {
            hasDotProductExt = false;
            hasCoopMatrixExt = false;
        }
    }

    private static bool QueryIntegerDotProductSupport(IntPtr physicalDevice)
    {
        try
        {
            var dotFeatures = new VulkanDriver.VkPhysicalDeviceShaderIntegerDotProductFeatures
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
                pNext = IntPtr.Zero,
                shaderIntegerDotProduct = 0
            };
            var features2 = new VulkanDriver.VkPhysicalDeviceFeatures2
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                pNext = new IntPtr(&dotFeatures)
            };
            VulkanDriver.GetPhysicalDeviceFeatures2(physicalDevice, ref features2);
            return dotFeatures.shaderIntegerDotProduct != 0;
        }
        catch
        {
            return false;
        }
    }

    private static bool QueryCooperativeMatrixSupport(IntPtr physicalDevice)
    {
        try
        {
            var coopFeatures = new VulkanDriver.VkPhysicalDeviceCooperativeMatrixFeaturesKHR
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                pNext = IntPtr.Zero,
                cooperativeMatrix = 0,
                cooperativeMatrixRobustBufferAccess = 0
            };
            var features2 = new VulkanDriver.VkPhysicalDeviceFeatures2
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                pNext = new IntPtr(&coopFeatures)
            };
            VulkanDriver.GetPhysicalDeviceFeatures2(physicalDevice, ref features2);
            return coopFeatures.cooperativeMatrix != 0;
        }
        catch
        {
            return false;
        }
    }

    public bool TryFindMemoryType(uint typeFilter, uint properties, uint avoidProperties, out uint memTypeIndex)
    {
        fixed (byte* pTypes = _memProps.memoryTypes)
        {
            var types = (VulkanDriver.VkMemoryType*)pTypes;
            for (uint i = 0; i < _memProps.memoryTypeCount; i++)
            {
                if ((typeFilter & (1u << (int)i)) != 0)
                {
                    uint flags = types[i].propertyFlags;
                    if ((flags & properties) == properties && (flags & avoidProperties) == 0)
                    {
                        memTypeIndex = i;
                        return true;
                    }
                }
            }
        }
        memTypeIndex = 0;
        return false;
    }

    public uint FindMemoryType(uint typeFilter, uint properties)
    {
        if (TryFindMemoryType(typeFilter, properties, 0u, out uint idx))
        {
            return idx;
        }
        throw new InvalidOperationException($"Failed to find suitable memory type for flags {properties}");
    }

    public void CreateBuffer(ulong size, uint usage, uint memProperties, out IntPtr buffer, out IntPtr memory)
        => CreateBufferPreferred(size, usage, out buffer, out memory, out _, (memProperties, 0u));

    public void CreateBufferPreferred(
        ulong size,
        uint usage,
        out IntPtr buffer,
        out IntPtr memory,
        params ReadOnlySpan<(uint Required, uint Avoid)> preferences)
        => CreateBufferPreferred(size, usage, out buffer, out memory, out _, preferences);

    public void CreateBufferPreferred(
        ulong size,
        uint usage,
        out IntPtr buffer,
        out IntPtr memory,
        out uint memoryPropertyFlags,
        params ReadOnlySpan<(uint Required, uint Avoid)> preferences)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var bufInfo = new VulkanDriver.VkBufferCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size = size,
            usage = usage,
            sharingMode = 0 // VK_SHARING_MODE_EXCLUSIVE
        };

        int res = VulkanDriver.CreateBuffer(_device, ref bufInfo, IntPtr.Zero, out buffer);
        if (res != VulkanDriver.VK_SUCCESS)
            throw new InvalidOperationException($"vkCreateBuffer failed for {size} bytes: code {res}");

        VulkanDriver.GetBufferMemoryRequirements(_device, buffer, out var memReqs);

        memory = IntPtr.Zero;
        memoryPropertyFlags = 0;
        int lastAllocRes = -1;
        bool matchedAny = false;
        uint testedMask = 0u;

        foreach (var (req, avoid) in preferences)
        {
            if (!TryFindMemoryType(memReqs.memoryTypeBits, req, avoid, out uint memTypeIndex))
            {
                continue;
            }

            uint bit = 1u << (int)memTypeIndex;
            if ((testedMask & bit) != 0u)
            {
                continue;
            }
            testedMask |= bit;
            matchedAny = true;

            var allocInfo = new VulkanDriver.VkMemoryAllocateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = memTypeIndex
            };

            lastAllocRes = VulkanDriver.AllocateMemory(_device, ref allocInfo, IntPtr.Zero, out memory);
            if (lastAllocRes == VulkanDriver.VK_SUCCESS)
            {
                fixed (byte* pTypes = _memProps.memoryTypes)
                {
                    memoryPropertyFlags = ((VulkanDriver.VkMemoryType*)pTypes)[memTypeIndex].propertyFlags;
                }
                break;
            }
            memory = IntPtr.Zero;
        }

        if (memory == IntPtr.Zero)
        {
            VulkanDriver.DestroyBuffer(_device, buffer, IntPtr.Zero);
            if (!matchedAny)
            {
                throw new InvalidOperationException($"Failed to find suitable memory type for buffer ({size} bytes).");
            }
            throw new InvalidOperationException($"vkAllocateMemory failed for {memReqs.size} bytes: code {lastAllocRes}");
        }

        int bindRes = VulkanDriver.BindBufferMemory(_device, buffer, memory, 0);
        if (bindRes != VulkanDriver.VK_SUCCESS)
        {
            VulkanDriver.FreeMemory(_device, memory, IntPtr.Zero);
            VulkanDriver.DestroyBuffer(_device, buffer, IntPtr.Zero);
            throw new InvalidOperationException($"vkBindBufferMemory failed: code {bindRes}");
        }
    }

    /// <summary>
    /// Allocates an optimal compute buffer for the active GPU topology:
    /// pure <c>DEVICE_LOCAL</c> GDDR6 VRAM on discrete GPUs (<see cref="PreferDeviceLocalStaging"/> == true),
    /// or cached zero-copy <c>DEVICE_LOCAL | HOST_VISIBLE</c> memory on UMA APUs.
    /// Automatically includes <c>TRANSFER_SRC</c> and <c>TRANSFER_DST</c> usage flags for staging copies.
    /// </summary>
    public void AllocateDeviceBuffer(
        ulong size,
        out IntPtr buffer,
        out IntPtr memory,
        out bool isHostVisible,
        uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    {
        AllocateDeviceBuffer(size, out buffer, out memory, out isHostVisible, out _, usage);
    }

    /// <summary>
    /// Allocates an optimal compute buffer for the active GPU topology and returns its raw Vulkan handles.
    /// </summary>
    public void AllocateDeviceBuffer(
        ulong size,
        out IntPtr buffer,
        out IntPtr memory,
        uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    {
        AllocateDeviceBuffer(size, out buffer, out memory, out _, out _, usage);
    }

    /// <summary>
    /// Allocates an optimal compute buffer wrapped in a disposable <see cref="VulkanDeviceBuffer"/> handle.
    /// </summary>
    public VulkanDeviceBuffer AllocateDeviceBufferHandle(
        ulong size,
        uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    {
        ulong allocBytes = Math.Max(size, 16UL);
        AllocateDeviceBuffer(allocBytes, out IntPtr buf, out IntPtr mem, out bool isHostVisible, out bool isDeviceLocal, usage);
        return new VulkanDeviceBuffer(this, buf, mem, allocBytes, isHostVisible, isDeviceLocal);
    }

    private void AllocateDeviceBuffer(
        ulong size,
        out IntPtr buffer,
        out IntPtr memory,
        out bool isHostVisible,
        out bool isDeviceLocal,
        uint usage)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        const uint hostCoherent = VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VulkanDriver.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const uint deviceLocalHostCoherent = VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | hostCoherent;
        ulong allocBytes = Math.Max(size, 16UL);
        uint fullUsage = usage | VulkanDriver.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VulkanDriver.VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        uint selectedFlags;
        if (PreferDeviceLocalStaging)
        {
            CreateBufferPreferred(
                allocBytes,
                fullUsage,
                out buffer,
                out memory,
                out selectedFlags,
                (VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
                (VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0u),
                (hostCoherent, 0u));
        }
        else
        {
            CreateBufferPreferred(
                allocBytes,
                fullUsage,
                out buffer,
                out memory,
                out selectedFlags,
                (deviceLocalHostCoherent, VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD),
                (deviceLocalHostCoherent, 0u),
                (hostCoherent, 0u));
        }

        isHostVisible = (selectedFlags & VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        isDeviceLocal = (selectedFlags & VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    }

    private void EnsureStagingBufferLocked(ulong bytes)
    {
        ulong wantStaging = Math.Min(bytes, DefaultMaxStagingBytes);
        if (_stagingBuffer != IntPtr.Zero && _stagingSize >= wantStaging)
        {
            return;
        }

        if (_stagingBuffer != IntPtr.Zero)
        {
            DestroyBuffer(_stagingBuffer, _stagingMemory);
            _stagingBuffer = IntPtr.Zero;
            _stagingMemory = IntPtr.Zero;
            _stagingSize = 0;
        }

        ulong allocStaging = Math.Max(wantStaging, DefaultMinStagingBytes);
        const uint hostCoherent = VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VulkanDriver.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        CreateBufferPreferred(
            allocStaging,
            VulkanDriver.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VulkanDriver.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            out _stagingBuffer,
            out _stagingMemory,
            (hostCoherent, VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            (hostCoherent, 0u));
        _stagingSize = allocStaging;
    }

    /// <summary>
    /// Uploads host memory into a Vulkan buffer using the reusable (up to 32 MB) staging buffer and <c>vkCmdCopyBuffer</c>.
    /// Enables full-bandwidth GDDR6 <c>DEVICE_LOCAL</c> residency on discrete GPUs.
    /// </summary>
    public void UploadViaStaging(IntPtr dstBuffer, IntPtr srcHost, ulong bytes, ulong dstOffset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (bytes == 0) return;
        if (dstBuffer == IntPtr.Zero) throw new ArgumentException("Destination buffer handle cannot be zero.", nameof(dstBuffer));
        if (srcHost == IntPtr.Zero) throw new ArgumentException("Source host pointer cannot be zero.", nameof(srcHost));

        lock (_stagingLock)
        {
            EnsureStagingBufferLocked(bytes);

            byte* srcPtr = (byte*)srcHost;
            ulong offset = 0;
            ulong remaining = bytes;
            IntPtr cb = _commandBuffer;

            while (remaining > 0)
            {
                ulong chunk = Math.Min(remaining, _stagingSize);
                CopyToBuffer(_stagingMemory, (IntPtr)(srcPtr + offset), chunk);

                lock (_queueLock)
                {
                    VulkanDriver.Check(VulkanDriver.ResetCommandBuffer(cb, 0), "vkResetCommandBuffer(staging upload)");
                    var begin = new VulkanDriver.VkCommandBufferBeginInfo
                    {
                        sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                    };
                    VulkanDriver.Check(VulkanDriver.BeginCommandBuffer(cb, ref begin), "vkBeginCommandBuffer(staging upload)");

                    var copy = new VulkanDriver.VkBufferCopy
                    {
                        srcOffset = 0,
                        dstOffset = dstOffset + offset,
                        size = chunk
                    };
                    VulkanDriver.CmdCopyBuffer(cb, _stagingBuffer, dstBuffer, 1, &copy);

                    VulkanDriver.Check(VulkanDriver.EndCommandBuffer(cb), "vkEndCommandBuffer(staging upload)");
                    SubmitAndWaitQueueLocked(cb, "staging upload");
                }

                offset += chunk;
                remaining -= chunk;
            }
        }
    }

    /// <summary>
    /// Uploads a typed host span into a Vulkan buffer via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void UploadViaStaging<T>(IntPtr dstBuffer, ReadOnlySpan<T> srcData, ulong dstOffset = 0) where T : unmanaged
    {
        if (srcData.IsEmpty) return;
        ulong bytes = (ulong)srcData.Length * (ulong)sizeof(T);
        fixed (T* pSrc = srcData)
        {
            UploadViaStaging(dstBuffer, (IntPtr)pSrc, bytes, dstOffset);
        }
    }

    /// <summary>
    /// Downloads data from a Vulkan buffer into host memory using the reusable (up to 32 MB) staging buffer and <c>vkCmdCopyBuffer</c>.
    /// Enables readback from pure <c>DEVICE_LOCAL</c> GDDR6 VRAM on discrete GPUs.
    /// </summary>
    public void DownloadViaStaging(IntPtr dstHost, IntPtr srcBuffer, ulong bytes, ulong srcOffset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (bytes == 0) return;
        if (srcBuffer == IntPtr.Zero) throw new ArgumentException("Source buffer handle cannot be zero.", nameof(srcBuffer));
        if (dstHost == IntPtr.Zero) throw new ArgumentException("Destination host pointer cannot be zero.", nameof(dstHost));

        lock (_stagingLock)
        {
            EnsureStagingBufferLocked(bytes);

            byte* dstPtr = (byte*)dstHost;
            ulong offset = 0;
            ulong remaining = bytes;
            IntPtr cb = _commandBuffer;

            while (remaining > 0)
            {
                ulong chunk = Math.Min(remaining, _stagingSize);

                lock (_queueLock)
                {
                    VulkanDriver.Check(VulkanDriver.ResetCommandBuffer(cb, 0), "vkResetCommandBuffer(staging download)");
                    var begin = new VulkanDriver.VkCommandBufferBeginInfo
                    {
                        sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                    };
                    VulkanDriver.Check(VulkanDriver.BeginCommandBuffer(cb, ref begin), "vkBeginCommandBuffer(staging download)");

                    var copy = new VulkanDriver.VkBufferCopy
                    {
                        srcOffset = srcOffset + offset,
                        dstOffset = 0,
                        size = chunk
                    };
                    VulkanDriver.CmdCopyBuffer(cb, srcBuffer, _stagingBuffer, 1, &copy);

                    VulkanDriver.Check(VulkanDriver.EndCommandBuffer(cb), "vkEndCommandBuffer(staging download)");
                    SubmitAndWaitQueueLocked(cb, "staging download");
                }

                CopyFromBuffer((IntPtr)(dstPtr + offset), _stagingMemory, chunk);

                offset += chunk;
                remaining -= chunk;
            }
        }
    }

    /// <summary>
    /// Downloads a Vulkan buffer into a typed host span via the reusable 32 MB staging buffer and <c>vkCmdCopyBuffer</c>.
    /// </summary>
    public void DownloadViaStaging<T>(Span<T> dstData, IntPtr srcBuffer, ulong srcOffset = 0) where T : unmanaged
    {
        if (dstData.IsEmpty) return;
        ulong bytes = (ulong)dstData.Length * (ulong)sizeof(T);
        fixed (T* pDst = dstData)
        {
            DownloadViaStaging((IntPtr)pDst, srcBuffer, bytes, srcOffset);
        }
    }

    /// <summary>
    /// Uploads host data into an allocated device buffer: uses direct zero-copy mapping when host-visible (UMA APU),
    /// or <see cref="UploadViaStaging(IntPtr, IntPtr, ulong, ulong)"/> when pure <c>DEVICE_LOCAL</c> (discrete GPU).
    /// </summary>
    public void UploadDeviceBuffer(IntPtr dstBuffer, IntPtr dstMemory, bool isHostVisible, IntPtr srcHost, ulong bytes, ulong dstOffset = 0)
    {
        if (bytes == 0) return;
        if (isHostVisible)
        {
            CopyToBuffer(dstMemory, srcHost, bytes, dstOffset);
        }
        else
        {
            UploadViaStaging(dstBuffer, srcHost, bytes, dstOffset);
        }
    }

    /// <summary>
    /// Downloads data from an allocated device buffer: uses direct zero-copy mapping when host-visible (UMA APU),
    /// or <see cref="DownloadViaStaging(IntPtr, IntPtr, ulong, ulong)"/> when pure <c>DEVICE_LOCAL</c> (discrete GPU).
    /// </summary>
    public void DownloadDeviceBuffer(IntPtr dstHost, IntPtr srcBuffer, IntPtr srcMemory, bool isHostVisible, ulong bytes, ulong srcOffset = 0)
    {
        if (bytes == 0) return;
        if (isHostVisible)
        {
            CopyFromBuffer(dstHost, srcMemory, bytes, srcOffset);
        }
        else
        {
            DownloadViaStaging(dstHost, srcBuffer, bytes, srcOffset);
        }
    }

    /// <summary>
    /// Releases the reusable staging buffer to reclaim host-visible memory immediately.
    /// </summary>
    public void ReleaseStagingBuffer()
    {
        lock (_stagingLock)
        {
            if (_stagingBuffer != IntPtr.Zero || _stagingMemory != IntPtr.Zero)
            {
                if (_device != IntPtr.Zero)
                {
                    VulkanDriver.DeviceWaitIdle(_device);
                    DestroyBuffer(_stagingBuffer, _stagingMemory);
                }
                _stagingBuffer = IntPtr.Zero;
                _stagingMemory = IntPtr.Zero;
                _stagingSize = 0;
            }
        }
    }

    public void DestroyBuffer(IntPtr buffer, IntPtr memory)
    {
        if (_device == IntPtr.Zero) return;
        if (buffer != IntPtr.Zero)
            VulkanDriver.DestroyBuffer(_device, buffer, IntPtr.Zero);
        if (memory != IntPtr.Zero)
            VulkanDriver.FreeMemory(_device, memory, IntPtr.Zero);
    }

    public void CopyToBuffer(IntPtr memory, IntPtr srcHost, ulong bytes)
        => CopyToBuffer(memory, srcHost, bytes, 0);

    public void CopyToBuffer(IntPtr memory, IntPtr srcHost, ulong bytes, ulong dstOffset)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (bytes == 0) return;
        int mapRes = VulkanDriver.MapMemory(_device, memory, dstOffset, bytes, 0, out IntPtr mapped);
        if (mapRes != VulkanDriver.VK_SUCCESS)
            throw new InvalidOperationException($"vkMapMemory failed: code {mapRes}");

        Buffer.MemoryCopy((void*)srcHost, (void*)mapped, bytes, bytes);
        VulkanDriver.UnmapMemory(_device, memory);
    }

    public void CopyFromBuffer(IntPtr dstHost, IntPtr memory, ulong bytes)
        => CopyFromBuffer(dstHost, memory, bytes, 0);

    public void CopyFromBuffer(IntPtr dstHost, IntPtr memory, ulong bytes, ulong srcOffset)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (bytes == 0) return;
        int mapRes = VulkanDriver.MapMemory(_device, memory, srcOffset, bytes, 0, out IntPtr mapped);
        if (mapRes != VulkanDriver.VK_SUCCESS)
            throw new InvalidOperationException($"vkMapMemory failed: code {mapRes}");

        Buffer.MemoryCopy((void*)mapped, (void*)dstHost, bytes, bytes);
        VulkanDriver.UnmapMemory(_device, memory);
    }

    /// <summary>
    /// Submits a recorded command buffer to the compute queue with <see cref="QueueFenceHandle"/> and waits on the fence.
    /// </summary>
    public void SubmitAndWait(IntPtr commandBuffer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (commandBuffer == IntPtr.Zero)
            throw new ArgumentException("Command buffer handle cannot be zero.", nameof(commandBuffer));

        lock (_queueLock)
        {
            SubmitAndWaitQueueLocked(commandBuffer, "submit");
        }
    }

    internal void RecordAndSubmitCompute(Action<IntPtr> recordCommands, string operationName = "compute")
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(recordCommands);

        lock (_queueLock)
        {
            IntPtr cb = _commandBuffer;
            VulkanDriver.Check(VulkanDriver.ResetCommandBuffer(cb, 0), $"vkResetCommandBuffer({operationName})");
            var begin = new VulkanDriver.VkCommandBufferBeginInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            };
            VulkanDriver.Check(VulkanDriver.BeginCommandBuffer(cb, ref begin), $"vkBeginCommandBuffer({operationName})");

            recordCommands(cb);

            VulkanDriver.Check(VulkanDriver.EndCommandBuffer(cb), $"vkEndCommandBuffer({operationName})");
            SubmitAndWaitQueueLocked(cb, operationName);
        }
    }

    private void SubmitAndWaitQueueLocked(IntPtr commandBuffer, string operationName)
    {
        if (_fenceInFlight && _queueFence != IntPtr.Zero)
        {
            VulkanDriver.Check(
                VulkanDriver.WaitForFences(_device, 1, ref _queueFence, VulkanDriver.VK_TRUE, ulong.MaxValue),
                "vkWaitForFences(pending)");
            VulkanDriver.Check(VulkanDriver.ResetFences(_device, 1, ref _queueFence), "vkResetFences(pending)");
            _fenceInFlight = false;
        }

        VulkanDriver.Check(VulkanDriver.ResetFences(_device, 1, ref _queueFence), $"vkResetFences({operationName})");
        var submit = new VulkanDriver.VkSubmitInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = new IntPtr(&commandBuffer)
        };

        _fenceInFlight = true;
        try
        {
            VulkanDriver.Check(VulkanDriver.QueueSubmit(_computeQueue, 1, ref submit, _queueFence), $"vkQueueSubmit({operationName})");
            VulkanDriver.Check(
                VulkanDriver.WaitForFences(_device, 1, ref _queueFence, VulkanDriver.VK_TRUE, ulong.MaxValue),
                $"vkWaitForFences({operationName})");
            VulkanDriver.Check(VulkanDriver.ResetFences(_device, 1, ref _queueFence), $"vkResetFences({operationName})");
        }
        finally
        {
            _fenceInFlight = false;
        }
    }

    public void Synchronize()
    {
        lock (_queueLock)
        {
            if (_device == IntPtr.Zero) return;

            if (_fenceInFlight && _queueFence != IntPtr.Zero)
            {
                try
                {
                    VulkanDriver.Check(
                        VulkanDriver.WaitForFences(_device, 1, ref _queueFence, VulkanDriver.VK_TRUE, ulong.MaxValue),
                        "vkWaitForFences(synchronize)");
                    VulkanDriver.Check(VulkanDriver.ResetFences(_device, 1, ref _queueFence), "vkResetFences(synchronize)");
                }
                finally
                {
                    _fenceInFlight = false;
                }
            }
            else if (_computeQueue != IntPtr.Zero)
            {
                VulkanDriver.Check(VulkanDriver.QueueWaitIdle(_computeQueue), "vkQueueWaitIdle");
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_device != IntPtr.Zero)
            {
                VulkanDriver.DeviceWaitIdle(_device);
                ReleaseStagingBuffer();
                if (_queueFence != IntPtr.Zero)
                {
                    VulkanDriver.DestroyFence(_device, _queueFence, IntPtr.Zero);
                    _queueFence = IntPtr.Zero;
                }
                if (_commandPool != IntPtr.Zero)
                {
                    VulkanDriver.DestroyCommandPool(_device, _commandPool, IntPtr.Zero);
                    _commandPool = IntPtr.Zero;
                }
                VulkanDriver.DestroyDevice(_device, IntPtr.Zero);
                _device = IntPtr.Zero;
            }

            if (_instance != IntPtr.Zero)
            {
                VulkanDriver.DestroyInstance(_instance, IntPtr.Zero);
                _instance = IntPtr.Zero;
            }

            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~VulkanContext()
    {
        Dispose();
    }
}
