using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Glacier.Gpu.Common;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Heterogeneous dual-GPU co-execution engine.
/// Coordinates simultaneous parallel execution across a discrete NVIDIA CUDA GPU and an AMD HIP GPU/APU.
/// Requires both hardware engines to initialize successfully.
/// </summary>
public sealed class HeterogeneousEngine : IGpuEngine
{
    private readonly NvidiaSassEngine _nvidia;
    private readonly AmdRdnaEngine _amd;
    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public NvidiaSassEngine Nvidia => _nvidia;
    public AmdRdnaEngine Amd => _amd;
    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;

    public HeterogeneousEngine(NvidiaSassEngine? nvidia = null, AmdRdnaEngine? amd = null)
    {
        _nvidia = nvidia ?? new NvidiaSassEngine();
        _amd = amd ?? new AmdRdnaEngine();
    }

    public void Initialize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;

        bool nvOk = false;
        bool amdOk = false;
        Exception? firstError = null;

        try
        {
            _nvidia.Initialize();
            nvOk = _nvidia.IsInitialized;
        }
        catch (Exception ex)
        {
            firstError = ex;
        }

        try
        {
            _amd.Initialize();
            amdOk = _amd.IsInitialized;
        }
        catch (Exception ex)
        {
            firstError ??= ex;
        }

        if (!nvOk || !amdOk)
        {
            _nvidia.Dispose();
            _amd.Dispose();
            throw new PlatformNotSupportedException(
                $"HeterogeneousEngine requires both NVIDIA CUDA and AMD HIP hardware drivers (NVIDIA={nvOk}, AMD={amdOk}).",
                firstError);
        }

        string desc = $"Dual-GPU Co-Execution: [NVIDIA: {_nvidia.DeviceInfo.DeviceName}] + [AMD: {_amd.DeviceInfo.DeviceName}]";

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: desc,
            DeviceType: GpuDeviceType.Unknown,
            Architecture: $"Heterogeneous ({_nvidia.DeviceInfo.Architecture} + {_amd.DeviceInfo.Architecture})",
            ComputeUnitsOrSms: _nvidia.DeviceInfo.ComputeUnitsOrSms + _amd.DeviceInfo.ComputeUnitsOrSms,
            TotalMemoryBytes: _nvidia.DeviceInfo.TotalMemoryBytes + _amd.DeviceInfo.TotalMemoryBytes,
            IsUnifiedMemory: _amd.DeviceInfo.IsUnifiedMemory,
            SupportsPersistentRingBuffer: _nvidia.DeviceInfo.SupportsPersistentRingBuffer
        );

        _initialized = true;
    }

    /// <summary>
    /// Simultaneously executes compute-dense SASS loops on NVIDIA dGPU and high-bandwidth memory sweeps on AMD APU.
    /// </summary>
    public (double nvidiaTflops, double amdBandwidthGbps, double totalMs) RunConcurrentWorkload(int nvElements = 2_097_152, int amdSizeMb = 128)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_initialized || !_nvidia.IsInitialized || !_amd.IsInitialized)
            throw new InvalidOperationException("Both NVIDIA and AMD engines must be initialized for concurrent workload.");

        var sw = Stopwatch.StartNew();

        var taskNvidia = Task.Run(() =>
        {
            Drivers.CuDriver.CtxSetCurrent(_nvidia.ContextHandle);
            return _nvidia.BenchmarkFmaCompute(N: nvElements, loopCount: 150);
        });

        var taskAmd = Task.Run(() => _amd.BenchmarkUnifiedBandwidth(sizeMb: amdSizeMb));

        Task.WaitAll(taskNvidia, taskAmd);
        sw.Stop();

        return (taskNvidia.Result.tflops, taskAmd.Result.bandwidthGbps, sw.Elapsed.TotalMilliseconds);
    }

    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _nvidia.Synchronize();
        _amd.Synchronize();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _initialized = false;
            _nvidia.Dispose();
            _amd.Dispose();
        }
    }
}
