using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Glacier.Gpu.Common;

namespace Glacier.Gpu.Engines;

/// <summary>
/// Heterogeneous dual-GPU co-execution engine.
/// Coordinates simultaneous parallel execution across discrete NVIDIA dGPU and integrated AMD APU.
/// </summary>
public sealed class HeterogeneousEngine : IGpuEngine
{
    private readonly NvidiaSassEngine? _nvidia;
    private readonly AmdRdnaEngine? _amd;
    private GpuDeviceInfo? _deviceInfo;
    private bool _initialized;
    private bool _disposed;

    public NvidiaSassEngine? Nvidia => _nvidia;
    public AmdRdnaEngine? Amd => _amd;
    public GpuDeviceInfo DeviceInfo => _deviceInfo ?? throw new InvalidOperationException("Engine not initialized.");
    public bool IsInitialized => _initialized && !_disposed;

    public HeterogeneousEngine(NvidiaSassEngine? nvidia = null, AmdRdnaEngine? amd = null)
    {
        _nvidia = nvidia ?? new NvidiaSassEngine();
        _amd = amd ?? new AmdRdnaEngine();
    }

    public void Initialize()
    {
        bool nvOk = false;
        bool amdOk = false;

        if (_nvidia != null)
        {
            try
            {
                _nvidia.Initialize();
                nvOk = _nvidia.IsInitialized;
            }
            catch { }
        }

        if (_amd != null)
        {
            try
            {
                _amd.Initialize();
                amdOk = _amd.IsInitialized;
            }
            catch { }
        }

        if (!nvOk && !amdOk)
        {
            throw new PlatformNotSupportedException("No supported GPU engines could be initialized.");
        }

        string desc = $"Dual-GPU Co-Execution: [NVIDIA: {(nvOk ? _nvidia!.DeviceInfo.DeviceName : "None")}] + [AMD: {(amdOk ? _amd!.DeviceInfo.DeviceName : "None")}]";

        _deviceInfo = new GpuDeviceInfo(
            DeviceName: desc,
            DeviceType: GpuDeviceType.Unknown,
            Architecture: "Heterogeneous (Ada Lovelace + RDNA 3.5)",
            ComputeUnitsOrSms: (nvOk ? _nvidia!.DeviceInfo.ComputeUnitsOrSms : 0) + (amdOk ? _amd!.DeviceInfo.ComputeUnitsOrSms : 0),
            TotalMemoryBytes: (nvOk ? _nvidia!.DeviceInfo.TotalMemoryBytes : 0) + (amdOk ? _amd!.DeviceInfo.TotalMemoryBytes : 0),
            IsUnifiedMemory: amdOk,
            SupportsPersistentRingBuffer: nvOk
        );

        _initialized = true;
    }

    /// <summary>
    /// Simultaneously executes compute-dense SASS loops on NVIDIA dGPU and high-bandwidth memory sweeps on AMD APU.
    /// </summary>
    public (double nvidiaTflops, double amdBandwidthGbps, double totalMs) RunConcurrentWorkload(int nvElements = 2_097_152, int amdSizeMb = 128)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_nvidia == null || !_nvidia.IsInitialized || _amd == null || !_amd.IsInitialized)
            throw new InvalidOperationException("Both NVIDIA and AMD engines must be initialized for concurrent workload.");

        var sw = Stopwatch.StartNew();

        var taskNvidia = Task.Run(() =>
        {
            Drivers.CuDriver.CtxSetCurrent(_nvidia.ContextHandle);
            return _nvidia.BenchmarkFmaCompute(N: nvElements, loopCount: 150);
        });

        var taskAmd = Task.Run(() =>
        {
            return _amd.BenchmarkUnifiedBandwidth(sizeMb: amdSizeMb);
        });

        Task.WaitAll(taskNvidia, taskAmd);
        sw.Stop();

        return (taskNvidia.Result.tflops, taskAmd.Result.bandwidthGbps, sw.Elapsed.TotalMilliseconds);
    }

    public void Synchronize()
    {
        _nvidia?.Synchronize();
        _amd?.Synchronize();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _nvidia?.Dispose();
            _amd?.Dispose();
        }
    }
}
