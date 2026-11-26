namespace Glacier.Gpu.Engines;

using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;

/// <summary>
/// Reusable NVIDIA CUDA Graph capture and replay executor.
/// Captures arbitrary single-stream or multi-stream kernel/copy sequences once via
/// <c>cuStreamBeginCapture_v2</c> / <c>cuStreamEndCapture</c> / <c>cuGraphInstantiateWithFlags</c>,
/// replays the entire DAG in a single <c>cuGraphLaunch</c> call (eliminating per-kernel WDDM/driver launch overhead),
/// and automatically falls back to direct stream execution if stream capture fails.
/// </summary>
public sealed class CudaGraphExecutor : IDisposable
{
    private readonly IntPtr _cudaContext;
    private readonly uint _captureMode;
    private readonly ulong _instantiateFlags;
    private readonly bool _ownsStream;
    private IntPtr _stream;
    private IntPtr _graphExec;
    private bool _captureFailed;
    private string? _lastCaptureError;
    private long _replayCount;
    private long _directExecutionCount;
    private bool _disposed;

    /// <summary>Gets the CUDA context handle bound to this executor.</summary>
    public IntPtr ContextHandle => _cudaContext;

    /// <summary>Gets the primary non-null CUDA stream used for capture and replay.</summary>
    public IntPtr StreamHandle => _stream;

    /// <summary>Gets the instantiated <c>CUgraphExec</c> handle, or <see cref="IntPtr.Zero"/> if not captured.</summary>
    public IntPtr GraphExecHandle => _graphExec;

    /// <summary>Gets a value indicating whether this executor created and owns <see cref="StreamHandle"/>.</summary>
    public bool OwnsStream => _ownsStream;

    /// <summary>Gets a value indicating whether an executable CUDA Graph is currently instantiated and ready for replay.</summary>
    public bool IsCaptured => _graphExec != IntPtr.Zero && !_disposed;

    /// <summary>Gets a value indicating whether graph capture previously failed and automatic direct-execution fallback is active.</summary>
    public bool IsFallbackActive => _captureFailed;

    /// <summary>Gets the error message from the most recent failed capture attempt, if any.</summary>
    public string? LastCaptureError => _lastCaptureError;

    /// <summary>Gets the number of times the captured CUDA Graph has been launched via <c>cuGraphLaunch</c>.</summary>
    public long ReplayCount => _replayCount;

    /// <summary>Gets the number of times direct stream execution fallback was used.</summary>
    public long DirectExecutionCount => _directExecutionCount;

    /// <summary>Gets a value indicating whether this executor has been disposed.</summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Initializes a new <see cref="CudaGraphExecutor"/> bound to an optional CUDA context and stream.
    /// When <paramref name="stream"/> is <see cref="IntPtr.Zero"/>, a dedicated non-blocking CUDA stream
    /// (<c>CU_STREAM_NON_BLOCKING</c>) is created and owned by this executor, because CUDA forbids graph capture
    /// on the legacy null stream.
    /// </summary>
    public CudaGraphExecutor(
        IntPtr cudaContext = default,
        IntPtr stream = default,
        uint captureMode = CuDriver.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        ulong instantiateFlags = 0)
    {
        if (!CuDriver.IsAvailable())
            throw new PlatformNotSupportedException("NVIDIA CUDA driver (nvcuda.dll / libcuda.so.1) is not available.");

        if (cudaContext != IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.CtxSetCurrent(cudaContext), "cuCtxSetCurrent");
            _cudaContext = cudaContext;
        }
        else
        {
            if (CuDriver.CtxGetCurrent(out IntPtr currentCtx) != 0 || currentCtx == IntPtr.Zero)
            {
                GpuBuffer<byte>.EnsureCudaContext();
                CuDriver.Check(CuDriver.CtxGetCurrent(out currentCtx), "cuCtxGetCurrent");
            }
            if (currentCtx == IntPtr.Zero)
                throw new InvalidOperationException("No active CUDA context could be established for CudaGraphExecutor.");
            _cudaContext = currentCtx;
        }

        _captureMode = captureMode;
        _instantiateFlags = instantiateFlags;

        if (stream != IntPtr.Zero)
        {
            _stream = stream;
            _ownsStream = false;
        }
        else
        {
            CuDriver.Check(CuDriver.StreamCreate(out _stream, CuDriver.CU_STREAM_NON_BLOCKING), "cuStreamCreate");
            _ownsStream = true;
        }
    }

    /// <summary>
    /// Initializes a new <see cref="CudaGraphExecutor"/> bound to an initialized <see cref="NvidiaSassEngine"/>.
    /// </summary>
    public CudaGraphExecutor(
        NvidiaSassEngine engine,
        IntPtr stream = default,
        uint captureMode = CuDriver.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        ulong instantiateFlags = 0)
        : this(ValidateEngine(engine), stream, captureMode, instantiateFlags)
    {
    }

    private static IntPtr ValidateEngine(NvidiaSassEngine engine)
    {
        ArgumentNullException.ThrowIfNull(engine);
        if (!engine.IsInitialized || engine.ContextHandle == IntPtr.Zero)
            throw new InvalidOperationException("NvidiaSassEngine must be initialized before creating a CudaGraphExecutor.");
        return engine.ContextHandle;
    }

    private void EnsureContext()
    {
        if (_cudaContext != IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.CtxSetCurrent(_cudaContext), "cuCtxSetCurrent");
        }
    }

    /// <summary>
    /// Captures <paramref name="recordSequence"/> on <see cref="StreamHandle"/> into an executable CUDA Graph.
    /// Note that CUDA stream capture records work without executing it; call <see cref="Replay"/> or
    /// <see cref="Execute"/> to launch the captured graph.
    /// Returns <c>true</c> on success, or <c>false</c> (activating fallback state) if capture fails.
    /// </summary>
    public bool TryCapture(Action<IntPtr> recordSequence)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(recordSequence);
        EnsureContext();

        InvalidateInternal();

        bool capturing = false;
        try
        {
            CuDriver.Check(CuDriver.StreamSynchronize(_stream), "cuStreamSynchronize(pre-capture)");
            CuDriver.Check(CuDriver.StreamBeginCapture(_stream, _captureMode), "cuStreamBeginCapture");
            capturing = true;

            recordSequence(_stream);

            CuDriver.Check(CuDriver.StreamEndCapture(_stream, out IntPtr graph), "cuStreamEndCapture");
            capturing = false;

            try
            {
                CuDriver.Check(CuDriver.GraphInstantiateWithFlags(out IntPtr graphExec, graph, _instantiateFlags), "cuGraphInstantiateWithFlags");
                _graphExec = graphExec;
            }
            finally
            {
                if (graph != IntPtr.Zero)
                {
                    CuDriver.GraphDestroy(graph);
                }
            }

            _captureFailed = false;
            _lastCaptureError = null;
            return true;
        }
        catch (Exception ex)
        {
            if (capturing)
            {
                // Always exit capture mode so the stream remains usable for direct fallback execution.
                if (CuDriver.StreamEndCapture(_stream, out IntPtr abortedGraph) == 0 && abortedGraph != IntPtr.Zero)
                {
                    CuDriver.GraphDestroy(abortedGraph);
                }
            }

            _captureFailed = true;
            _lastCaptureError = ex.Message;
            return false;
        }
    }

    /// <summary>
    /// Executes <paramref name="recordSequence"/>: captures &amp; instantiates the CUDA Graph on first call,
    /// replays the captured graph in 1 <c>cuGraphLaunch</c> on this and all subsequent calls,
    /// and automatically falls back to direct stream execution if capture fails.
    /// </summary>
    public void Execute(Action<IntPtr> recordSequence, bool synchronize = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(recordSequence);
        EnsureContext();

        if (_graphExec != IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.GraphLaunch(_graphExec, _stream), "cuGraphLaunch");
            _replayCount++;
        }
        else if (!_captureFailed && TryCapture(recordSequence))
        {
            CuDriver.Check(CuDriver.GraphLaunch(_graphExec, _stream), "cuGraphLaunch");
            _replayCount++;
        }
        else
        {
            recordSequence(_stream);
            _directExecutionCount++;
        }

        if (synchronize)
        {
            CuDriver.Check(CuDriver.StreamSynchronize(_stream), "cuStreamSynchronize");
        }
    }

    /// <summary>
    /// Replays an already-captured CUDA Graph in a single <c>cuGraphLaunch</c> call.
    /// </summary>
    public void Replay(bool synchronize = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_graphExec == IntPtr.Zero)
            throw new InvalidOperationException("No CUDA Graph has been captured yet. Call TryCapture or Execute first.");

        EnsureContext();
        CuDriver.Check(CuDriver.GraphLaunch(_graphExec, _stream), "cuGraphLaunch");
        _replayCount++;

        if (synchronize)
        {
            CuDriver.Check(CuDriver.StreamSynchronize(_stream), "cuStreamSynchronize");
        }
    }

    /// <summary>
    /// Forks an auxiliary CUDA stream from <see cref="StreamHandle"/> using <paramref name="forkEvent"/>
    /// so parallel multi-stream work is cleanly captured into the same CUDA Graph DAG.
    /// </summary>
    public void ForkStream(IntPtr auxStream, IntPtr forkEvent)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (auxStream == IntPtr.Zero) throw new ArgumentException("Auxiliary stream cannot be zero.", nameof(auxStream));
        if (forkEvent == IntPtr.Zero) throw new ArgumentException("Fork event cannot be zero.", nameof(forkEvent));

        CuDriver.Check(CuDriver.EventRecord(forkEvent, _stream), "cuEventRecord(fork)");
        CuDriver.Check(CuDriver.StreamWaitEvent(auxStream, forkEvent, 0), "cuStreamWaitEvent(fork)");
    }

    /// <summary>
    /// Joins an auxiliary CUDA stream back into <see cref="StreamHandle"/> using <paramref name="joinEvent"/>
    /// before ending CUDA Graph capture.
    /// </summary>
    public void JoinStream(IntPtr auxStream, IntPtr joinEvent)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (auxStream == IntPtr.Zero) throw new ArgumentException("Auxiliary stream cannot be zero.", nameof(auxStream));
        if (joinEvent == IntPtr.Zero) throw new ArgumentException("Join event cannot be zero.", nameof(joinEvent));

        CuDriver.Check(CuDriver.EventRecord(joinEvent, auxStream), "cuEventRecord(join)");
        CuDriver.Check(CuDriver.StreamWaitEvent(_stream, joinEvent, 0), "cuStreamWaitEvent(join)");
    }

    /// <summary>
    /// Synchronizes <see cref="StreamHandle"/> until all queued graph launches or direct operations complete.
    /// </summary>
    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureContext();
        CuDriver.Check(CuDriver.StreamSynchronize(_stream), "cuStreamSynchronize");
    }

    /// <summary>
    /// Destroys the currently instantiated CUDA Graph (if any) and resets fallback state so a new sequence can be captured.
    /// </summary>
    public void Invalidate()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureContext();
        InvalidateInternal();
        _captureFailed = false;
        _lastCaptureError = null;
    }

    private void InvalidateInternal()
    {
        if (_graphExec != IntPtr.Zero)
        {
            if (_stream != IntPtr.Zero)
            {
                CuDriver.StreamSynchronize(_stream);
            }
            CuDriver.GraphExecDestroy(_graphExec);
            _graphExec = IntPtr.Zero;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (CuDriver.IsAvailable())
            {
                if (_cudaContext != IntPtr.Zero)
                {
                    CuDriver.CtxSetCurrent(_cudaContext);
                }

                InvalidateInternal();

                if (_ownsStream && _stream != IntPtr.Zero)
                {
                    CuDriver.StreamSynchronize(_stream);
                    CuDriver.StreamDestroy(_stream);
                }
            }
            _stream = IntPtr.Zero;
        }
    }
}
