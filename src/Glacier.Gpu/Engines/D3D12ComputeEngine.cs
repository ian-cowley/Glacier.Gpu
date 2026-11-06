using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Threading;
using Glacier.Gpu.Common;
using Glacier.Gpu.RingBuffer;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Hardware compute execution engine driven directly via bare-metal Direct3D 12 Compute (d3d12.dll).
/// Native execution on AMD Radeon 680M/780M/890M APUs, Intel Arc/Xe, and RDNA/NVIDIA dGPUs on Windows.
/// Isolates all Windows DXGI/D3D12 calls behind NoInlining boundaries for safe cross-platform assembly loading.
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

    private ID3D12RootSignature? _computeRootSig;
    private ID3D12PipelineState? _psoVecAdd;
    private ID3D12PipelineState? _psoVecFma;

    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;

    [SupportedOSPlatform("windows")]
    public ID3D12Device Device => _device ?? throw new InvalidOperationException("Device not created.");

    [SupportedOSPlatform("windows")]
    public ID3D12CommandQueue Queue => _queue ?? throw new InvalidOperationException("Queue not created.");

    public D3D12ComputeEngine(int adapterIndex = -1)
    {
        _adapterIndex = adapterIndex;
    }

    public void Initialize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;

        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("Direct3D 12 Compute is only supported on Windows.");

        InitializeWindows();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void InitializeWindows()
    {
        _factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();

        IDXGIAdapter1? chosenAdapter = null;

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
                    break;
                }
                a.Dispose();
                continue;
            }

            if (desc.Description.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
            {
                chosenAdapter?.Dispose();
                chosenAdapter = a;
                break;
            }

            if (chosenAdapter == null)
            {
                chosenAdapter = a;
            }
            else
            {
                a.Dispose();
            }
        }

        if (chosenAdapter == null)
        {
            _factory.Dispose();
            _factory = null;
            throw new PlatformNotSupportedException("No hardware DirectX 12 compute adapters found on this system.");
        }

        _adapter = chosenAdapter;
        var adapterDesc = _adapter.Description1;
        string devName = adapterDesc.Description;

        var hr = D3D12.D3D12CreateDevice(_adapter, FeatureLevel.Level_11_0, out _device);
        if (!hr.Success || _device == null)
            throw new InvalidOperationException($"Failed to create Direct3D 12 device for '{devName}': {hr}");

        var queueDesc = new CommandQueueDescription(CommandListType.Compute);
        _queue = _device.CreateCommandQueue(queueDesc);

        _cmdAlloc = _device.CreateCommandAllocator(CommandListType.Compute);
        _cmdList = _device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, _cmdAlloc);
        _cmdList.Close();

        _fence = _device.CreateFence(0);
        _fenceValue = 0;
        _fenceEvent = new AutoResetEvent(false);

        string lowerName = devName.ToLowerInvariant();
        bool isApu = lowerName.Contains("890m") || lowerName.Contains("880m") ||
                     lowerName.Contains("780m") || lowerName.Contains("760m") ||
                     lowerName.Contains("680m") || lowerName.Contains("660m") ||
                     lowerName.Contains("graphics") || lowerName.Contains("apu") ||
                     ((ulong)adapterDesc.DedicatedVideoMemory == 0 && (ulong)adapterDesc.SharedSystemMemory > 0);

        string arch = lowerName.Contains("radeon")
            ? AmdRdnaEngine.ResolveArchitecture(devName, isApu)
            : "DirectX 12 Compute";

        long totalMem = (long)(ulong)adapterDesc.DedicatedVideoMemory;
        if (totalMem == 0 || isApu)
        {
            totalMem = (long)(ulong)adapterDesc.SharedSystemMemory;
        }

        GpuDeviceType devType;
        if (adapterDesc.VendorId == 0x1002 || lowerName.Contains("radeon"))
            devType = isApu ? GpuDeviceType.AmdIntegratedApu : GpuDeviceType.AmdDiscrete;
        else if (adapterDesc.VendorId == 0x8086)
            devType = isApu ? GpuDeviceType.IntelIntegrated : GpuDeviceType.IntelDiscrete;
        else if (adapterDesc.VendorId == 0x10DE)
            devType = GpuDeviceType.NvidiaDiscrete;
        else
            devType = GpuDeviceType.DirectMlGeneric;

        // Direct3D 12 dispatches via WDDM command lists rather than persistent GPU host-polling megakernels.
        _deviceInfo = new GpuDeviceInfo(
            DeviceName: devName,
            DeviceType: devType,
            Architecture: arch,
            ComputeUnitsOrSms: 0,
            TotalMemoryBytes: totalMem,
            IsUnifiedMemory: isApu,
            SupportsPersistentRingBuffer: false
        );

        _initialized = true;
    }

    /// <summary>
    /// Allocates contiguous zero-copy host-accessible GPU buffer mapped directly into CPU and GPU address spaces.
    /// </summary>
    public UnifiedMemoryBlock<T> AllocateZeroCopy<T>(int count) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("Direct3D 12 Compute is only supported on Windows.");
        return AllocateZeroCopyWindows<T>(count);
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private UnifiedMemoryBlock<T> AllocateZeroCopyWindows<T>(int count) where T : unmanaged
    {
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
    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
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

    /// <summary>
    /// Direct3D 12 uses WDDM command-list submission and does not host persistent GPU polling megakernels.
    /// Always throws <see cref="NotSupportedException"/>; check <c>DeviceInfo.SupportsPersistentRingBuffer</c> first.
    /// </summary>
    public PersistentRingBuffer GetOrCreateRingBuffer(int threadCount = 1024)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        throw new NotSupportedException(
            "Direct3D 12 Compute uses WDDM command-queue submission and does not support persistent GPU host-polling ring buffer workers. " +
            "Use NvidiaSassEngine for persistent ring buffer execution, or MeasureCommandListDispatchLatency for D3D12 command-list benchmarks.");
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void EnsureComputePipelinesWindows()
    {
        if (_computeRootSig != null) return;

        var rootParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
        };

        var rootSigDesc = new RootSignatureDescription(RootSignatureFlags.None, rootParams);
        _computeRootSig = _device!.CreateRootSignature(rootSigDesc, RootSignatureVersion.Version1);

        const string hlslVecAdd = @"
            cbuffer Params : register(b0)
            {
                uint count;
                float scalar;
            };
            RWStructuredBuffer<float> bufA : register(u0);
            RWStructuredBuffer<float> bufB : register(u1);
            RWStructuredBuffer<float> bufC : register(u2);

            [numthreads(64, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID)
            {
                if (id.x < count)
                {
                    bufC[id.x] = bufA[id.x] + bufB[id.x];
                }
            }
        ";

        const string hlslVecFma = @"
            cbuffer Params : register(b0)
            {
                uint count;
                float scalar;
            };
            RWStructuredBuffer<float> bufA : register(u0);
            RWStructuredBuffer<float> bufB : register(u1);
            RWStructuredBuffer<float> bufC : register(u2);

            [numthreads(64, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID)
            {
                if (id.x < count)
                {
                    bufC[id.x] = bufA[id.x] * scalar + bufB[id.x];
                }
            }
        ";

        var blobVecAdd = Compiler.Compile(hlslVecAdd, "main", "d3d12_vec_add.hlsl", "cs_5_0");
        _psoVecAdd = _device.CreateComputePipelineState(new ComputePipelineStateDescription
        {
            RootSignature = _computeRootSig,
            ComputeShader = blobVecAdd
        });

        var blobVecFma = Compiler.Compile(hlslVecFma, "main", "d3d12_vec_fma.hlsl", "cs_5_0");
        _psoVecFma = _device.CreateComputePipelineState(new ComputePipelineStateDescription
        {
            RootSignature = _computeRootSig,
            ComputeShader = blobVecFma
        });
    }

    /// <summary>
    /// Measures honest Direct3D 12 command-list recording + ExecuteCommandList submission latency and fence-synchronized GPU turnaround.
    /// </summary>
    public (double dispatchUs, double roundTripUs, bool verified, float sampleVal) MeasureCommandListDispatchLatency(int iterations = 1_000, int elementCount = 256)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("Direct3D 12 Compute is only supported on Windows.");
        return MeasureCommandListDispatchLatencyWindows(iterations, elementCount);
    }

    /// <summary>
    /// Retained for benchmark signature compatibility; delegates to <see cref="MeasureCommandListDispatchLatency"/>.
    /// Note that Direct3D 12 executes WDDM command lists rather than persistent GPU polling workers.
    /// </summary>
    [Obsolete("D3D12 executes WDDM command lists rather than a persistent GPU ring worker. Use MeasureCommandListDispatchLatency instead.")]
    public (double dispatchUs, double roundTripUs, bool verified, float sampleVal) MeasureRingBufferLatency(int iterations = 1_000, int elementCount = 256)
        => MeasureCommandListDispatchLatency(iterations, elementCount);

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private (double dispatchUs, double roundTripUs, bool verified, float sampleVal) MeasureCommandListDispatchLatencyWindows(int iterations, int elementCount)
    {
        if (_device == null || _queue == null || _cmdList == null || _cmdAlloc == null)
            throw new InvalidOperationException("Device not initialized.");

        EnsureComputePipelinesWindows();

        int n = Math.Max(1, elementCount);
        ulong bufferBytes = (ulong)(n * sizeof(float));

        using var d_a = AllocateDeviceBuffer(bufferBytes);
        using var d_b = AllocateDeviceBuffer(bufferBytes);
        using var d_c = AllocateDeviceBuffer(bufferBytes);

        float[] initA = new float[n];
        float[] initB = new float[n];
        Array.Fill(initA, 10.0f);
        Array.Fill(initB, 20.0f);

        using var upA = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Upload), HeapFlags.None,
            ResourceDescription.Buffer(bufferBytes), ResourceStates.GenericRead);
        using var upB = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Upload), HeapFlags.None,
            ResourceDescription.Buffer(bufferBytes), ResourceStates.GenericRead);

        void* pUpA = null;
        void* pUpB = null;
        upA.Map(0, null, &pUpA);
        upB.Map(0, null, &pUpB);
        fixed (float* pInitA = initA, pInitB = initB)
        {
            Buffer.MemoryCopy(pInitA, pUpA, bufferBytes, bufferBytes);
            Buffer.MemoryCopy(pInitB, pUpB, bufferBytes, bufferBytes);
        }
        upA.Unmap(0);
        upB.Unmap(0);

        _cmdAlloc.Reset();
        _cmdList.Reset(_cmdAlloc, null);
        _cmdList.CopyResource(d_a, upA);
        _cmdList.CopyResource(d_b, upB);
        _cmdList.Close();
        _queue.ExecuteCommandList(_cmdList);
        SynchronizeWindows();

        int roundTripIters = Math.Clamp(iterations, 1, 1_000);
        uint dispatchGroups = (uint)((n + 63) / 64);
        uint* pConstsAdd = stackalloc uint[2];
        pConstsAdd[0] = (uint)n;
        float zero = 0.0f;
        pConstsAdd[1] = *(uint*)&zero;

        long submitTicks = 0;
        var swRoundTrip = Stopwatch.StartNew();
        for (int i = 0; i < roundTripIters; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            _cmdAlloc.Reset();
            _cmdList.Reset(_cmdAlloc, _psoVecAdd);
            _cmdList.SetComputeRootSignature(_computeRootSig!);
            _cmdList.SetComputeRoot32BitConstants(0, 2, (IntPtr)pConstsAdd, 0);
            _cmdList.SetComputeRootUnorderedAccessView(1, d_a.GPUVirtualAddress);
            _cmdList.SetComputeRootUnorderedAccessView(2, d_b.GPUVirtualAddress);
            _cmdList.SetComputeRootUnorderedAccessView(3, d_c.GPUVirtualAddress);
            _cmdList.Dispatch(dispatchGroups, 1, 1);
            _cmdList.Close();
            _queue.ExecuteCommandList(_cmdList);
            submitTicks += Stopwatch.GetTimestamp() - t0;

            SynchronizeWindows();
        }
        swRoundTrip.Stop();

        double dispatchUs = Stopwatch.GetElapsedTime(0, submitTicks).TotalMicroseconds / roundTripIters;
        double roundTripUs = swRoundTrip.Elapsed.TotalMicroseconds / roundTripIters;

        // Verify VectorFma on GPU: C = A * 2.0f + B = 10 * 2 + 20 = 40
        uint* pConstsFma = stackalloc uint[2];
        pConstsFma[0] = (uint)n;
        float fmaScalar = 2.0f;
        pConstsFma[1] = *(uint*)&fmaScalar;

        using var readback = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Readback), HeapFlags.None,
            ResourceDescription.Buffer(bufferBytes), ResourceStates.CopyDest);

        _cmdAlloc.Reset();
        _cmdList.Reset(_cmdAlloc, _psoVecFma);
        _cmdList.SetComputeRootSignature(_computeRootSig!);
        _cmdList.SetComputeRoot32BitConstants(0, 2, (IntPtr)pConstsFma, 0);
        _cmdList.SetComputeRootUnorderedAccessView(1, d_a.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(2, d_b.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(3, d_c.GPUVirtualAddress);
        _cmdList.Dispatch(dispatchGroups, 1, 1);

        _cmdList.ResourceBarrierTransition(d_c, ResourceStates.Common, ResourceStates.CopySource);
        _cmdList.CopyResource(readback, d_c);
        _cmdList.ResourceBarrierTransition(d_c, ResourceStates.CopySource, ResourceStates.Common);
        _cmdList.Close();

        _queue.ExecuteCommandList(_cmdList);
        SynchronizeWindows();

        void* pRead = null;
        readback.Map(0, null, &pRead);
        float* pOut = (float*)pRead;
        float sampleVal = pOut[0];
        bool verified = Math.Abs(sampleVal - 40.0f) < 1e-3f && Math.Abs(pOut[n - 1] - 40.0f) < 1e-3f;
        readback.Unmap(0);

        return (dispatchUs, roundTripUs, verified, sampleVal);
    }

    public void Synchronize()
    {
        if (!OperatingSystem.IsWindows() || _disposed) return;
        SynchronizeWindows();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void SynchronizeWindows()
    {
        if (_queue == null || _fence == null || _fenceEvent == null) return;

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
            if (OperatingSystem.IsWindows())
            {
                DisposeWindows();
            }
        }
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void DisposeWindows()
    {
        try { SynchronizeWindows(); } catch { }

        _psoVecAdd?.Dispose();
        _psoVecFma?.Dispose();
        _computeRootSig?.Dispose();

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
