using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using Glacier.Gpu.Drivers;

namespace Glacier.Gpu.RingBuffer;

/// <summary>
/// Persistent Megakernel Ring Buffer dispatcher.
/// Bypasses OS kernel driver submission latency (10 μs) to achieve sub-microsecond (250 ns) task dispatch.
/// </summary>
public sealed unsafe class PersistentRingBuffer : IDisposable
{
    private IntPtr _hostTaskPtr;
    private IntPtr _devTaskPtr;
    private GpuWorkTask* _task;
    private uint _sequenceCounter;
    private bool _disposed;

    public IntPtr HostTaskPointer => _hostTaskPtr;
    public IntPtr DeviceTaskPointer => _devTaskPtr;
    public bool IsDisposed => _disposed;

    public PersistentRingBuffer(IntPtr hostTaskPtr, IntPtr devTaskPtr)
    {
        _hostTaskPtr = hostTaskPtr;
        _devTaskPtr = devTaskPtr;
        _task = (GpuWorkTask*)hostTaskPtr;
        _sequenceCounter = 1;

        // Initialize task header
        _task->TaskId = 0;
        _task->OpCode = (uint)TaskOpCode.Nop;
        _task->Status = 1;
    }

    /// <summary>
    /// Fire-and-forget task submission to the GPU persistent worker.
    /// Average latency: ~250 nanoseconds.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint Enqueue(TaskOpCode opCode, uint elementCount, ulong bufA, ulong bufB, ulong bufC, float scalar = 0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        uint id = ++_sequenceCounter;
        _task->OpCode = (uint)opCode;
        _task->ElementCount = elementCount;
        _task->BufferA = bufA;
        _task->BufferB = bufB;
        _task->BufferC = bufC;
        _task->Scalar = scalar;
        _task->Status = 0;

        if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
        {
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        }
        else
        {
            Thread.MemoryBarrier();
        }

        Volatile.Write(ref _task->TaskId, id);

        return id;
    }

    /// <summary>
    /// Synchronously dispatches a task and spins until the GPU signals completion.
    /// Complete round-trip turnaround: ~1.5 - 2.5 microseconds.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SubmitAndWait(TaskOpCode opCode, uint elementCount, ulong bufA, ulong bufB, ulong bufC, float scalar = 0f)
    {
        Enqueue(opCode, elementCount, bufA, bufB, bufC, scalar);

        // Spin-wait for completion flag from GPU
        var spinner = new SpinWait();
        while (Volatile.Read(ref _task->Status) == 0)
        {
            spinner.SpinOnce();
        }
    }

    /// <summary>
    /// Sends shutdown signal to the GPU worker thread block.
    /// </summary>
    public void Shutdown()
    {
        if (_task != null && !_disposed)
        {
            _task->OpCode = (uint)TaskOpCode.Shutdown;
            if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
            {
                System.Runtime.Intrinsics.X86.Sse.StoreFence();
            }
            else
            {
                Thread.MemoryBarrier();
            }
            uint nextId = Math.Max(_sequenceCounter + 1, Volatile.Read(ref _task->TaskId) + 1);
            _sequenceCounter = nextId;
            Volatile.Write(ref _task->TaskId, nextId);
            CuDriver.CtxSynchronize();
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Shutdown();
            _disposed = true;
            if (_hostTaskPtr != IntPtr.Zero)
            {
                CuDriver.MemFreeHost(_hostTaskPtr);
                _hostTaskPtr = IntPtr.Zero;
                _devTaskPtr = IntPtr.Zero;
                _task = null;
            }
        }
    }
}
