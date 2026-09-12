using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Gpu.Common;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Hardware compute execution engine driven directly via bare-metal Direct3D 12 Compute (d3d12.dll).
/// Native execution on AMD Radeon 680M/780M/890M APUs and RDNA dGPUs with zero external C++ runtimes.
/// Operates on unified system memory with zero PCIe copy overhead.
/// </summary>
public sealed unsafe class D3D12ComputeEngine : IGpuEngine
{
    private readonly int _adapterIndex;
    private IDXGIFactory4? _factory;
    private IDXGIAdapter1? _adapter;
    private ID3D12Device? _device;
    private ID3D12CommandQueue? _queue;
    private ID3D12CommandAllocator? _cmdAlloc;
    private ID3D12GraphicsCommandList? _cmdList;
    private ID3D12Fence? _fence;
    private ulong _fenceValue;
    private AutoResetEvent? _fenceEvent;

    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;
    public ID3D12Device Device => _device ?? throw new InvalidOperationException("Device not created.");
    public ID3D12CommandQueue Queue => _queue ?? throw new InvalidOperationException("Queue not created.");

    public D3D12ComputeEngine(int adapterIndex = -1)
    {
        _adapterIndex = adapterIndex;
    }

    public void Initialize()
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("Direct3D 12 Compute is only supported on Windows.");

        _factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();

        // Enumerate adapters to select target GPU
        IDXGIAdapter1? chosenAdapter = null;
        int chosenIdx = -1;

        for (uint i = 0; _factory.EnumAdapters1(i, out IDXGIAdapter1 a).Success; i++)
        {
            var desc = a.Description1;
            if ((desc.Flags & AdapterFlags.Software) != 0)
            {
                a.Dispose();
                continue;
            }

            if (_adapterIndex >= 0)
            {
                if ((int)i == _adapterIndex)
                {
                    chosenAdapter = a;
                    chosenIdx = (int)i;
                    break;
                }
                a.Dispose();
                continue;
            }

            // Default selection: Prefer AMD Radeon GPU, otherwise first hardware GPU
            if (desc.Description.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
            {
                chosenAdapter = a;
                chosenIdx = (int)i;
                break;
            }

            if (chosenAdapter == null)
            {
                chosenAdapter = a;
                chosenIdx = (int)i;
            }
            else
            {
                a.Dispose();
            }
        }

        if (chosenAdapter == null)
            throw new InvalidOperationException("No hardware DirectX 12 compute adapters found on this system.");

        _adapter = chosenAdapter;
        var adapterDesc = _adapter.Description1;
        string devName = adapterDesc.Description;

        // Create D3D12 Device
        var hr = D3D12.D3D12CreateDevice(_adapter, FeatureLevel.Level_11_0, out _device);
        if (!hr.Success || _device == null)
            throw new InvalidOperationException($"Failed to create Direct3D 12 device for '{devName}': {hr}");

        // Create Compute Command Queue
        var queueDesc = new CommandQueueDescription(CommandListType.Compute);
        _queue = _device.CreateCommandQueue(queueDesc);

        // Create Command Allocator & List
        _cmdAlloc = _device.CreateCommandAllocator(CommandListType.Compute);
        _cmdList = _device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, _cmdAlloc);
        _cmdList.Close();

        // Create Fence for CPU-GPU synchronization
        _fence = _device.CreateFence(0);
        _fenceValue = 0;
        _fenceEvent = new AutoResetEvent(false);

        // Determine GPU architecture classification
        string lowerName = devName.ToLowerInvariant();
        bool isApu = lowerName.Contains("890m") || lowerName.Contains("880m") ||
                     lowerName.Contains("780m") || lowerName.Contains("760m") ||
                     lowerName.Contains("680m") || lowerName.Contains("660m") ||
                     lowerName.Contains("graphics") || lowerName.Contains("apu");

        string arch;
        if (lowerName.Contains("890m") || lowerName.Contains("880m"))
            arch = "RDNA 3.5 (gfx1150)";
        else if (lowerName.Contains("780m") || lowerName.Contains("760m") || lowerName.Contains("740m"))
            arch = "RDNA 3.0 (gfx1103)";
        else if (lowerName.Contains("680m") || lowerName.Contains("660m"))
            arch = "RDNA 2.0 (gfx1035)";
        else if (lowerName.Contains("radeon"))
            arch = isApu ? "RDNA APU" : "RDNA dGPU";
        else
            arch = "DirectX 12 Compute";

        long totalMem = (long)(ulong)adapterDesc.DedicatedVideoMemory;
        if (totalMem == 0 || isApu)
        {
            totalMem = (long)(ulong)adapterDesc.SharedSystemMemory;
        }

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: devName,
            DeviceType: isApu ? GpuDeviceType.AmdIntegratedApu : (lowerName.Contains("radeon") ? GpuDeviceType.AmdDiscrete : GpuDeviceType.DirectMlGeneric),
            Architecture: arch,
            ComputeUnitsOrSms: isApu ? 12 : 0,
            TotalMemoryBytes: totalMem,
            IsUnifiedMemory: isApu,
            SupportsPersistentRingBuffer: false
        );

        _initialized = true;
    }

    /// <summary>
    /// Allocates contiguous zero-copy host-accessible GPU buffer mapped directly into CPU and GPU address spaces.
    /// 0 ns PCIe copy penalty on unified memory architectures (UMA).
    /// </summary>
    public UnifiedMemoryBlock<T> AllocateZeroCopy<T>(int count) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_device == null) throw new InvalidOperationException("Device not initialized.");

        int byteSize = count * sizeof(T);
        var uploadDesc = ResourceDescription.Buffer((ulong)byteSize);
        var uploadHeap = new HeapProperties(HeapType.Upload);

        var resource = _device.CreateCommittedResource(
            uploadHeap,
            HeapFlags.None,
            uploadDesc,
            ResourceStates.GenericRead);

        void* pHostData = null;
        resource.Map(0, null, &pHostData);

        IntPtr hostPtr = (IntPtr)pHostData;
        IntPtr devPtr = (IntPtr)resource.GPUVirtualAddress;

        return new UnifiedMemoryBlock<T>(hostPtr, devPtr, count, _ =>
        {
            resource.Unmap(0);
            resource.Dispose();
        });
    }

    /// <summary>
    /// Allocates GPU-only device buffer in high-speed local memory / GPU cache.
    /// </summary>
    public ID3D12Resource AllocateDeviceBuffer(ulong byteSize, ResourceFlags flags = ResourceFlags.AllowUnorderedAccess)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_device == null) throw new InvalidOperationException("Device not initialized.");

        var desc = ResourceDescription.Buffer(byteSize, flags);
        var defaultHeap = new HeapProperties(HeapType.Default);

        return _device.CreateCommittedResource(
            defaultHeap,
            HeapFlags.None,
            desc,
            ResourceStates.Common);
    }

    public void Synchronize()
    {
        if (_queue == null || _fence == null || _fenceEvent == null || _disposed) return;

        _fenceValue++;
        _queue.Signal(_fence, _fenceValue);

        if (_fence.CompletedValue < _fenceValue)
        {
            _fence.SetEventOnCompletion(_fenceValue, _fenceEvent);
            _fenceEvent.WaitOne();
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _initialized = false;

            Synchronize();

            _fenceEvent?.Dispose();
            _fence?.Dispose();
            _cmdList?.Dispose();
            _cmdAlloc?.Dispose();
            _queue?.Dispose();
            _device?.Dispose();
            _adapter?.Dispose();
            _factory?.Dispose();
        }
    }
}
