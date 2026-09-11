using System;
using System.Runtime.CompilerServices;

namespace Glacier.Gpu.Common;

/// <summary>
/// Contiguous unmanaged memory block mapped simultaneously into CPU and GPU address spaces.
/// Enables true zero-copy data exchange with zero PCIe staging copies.
/// </summary>
public sealed unsafe class UnifiedMemoryBlock<T> : IDisposable where T : unmanaged
{
    private IntPtr _hostPtr;
    private IntPtr _devPtr;
    private readonly nuint _sizeInBytes;
    private readonly int _elementCount;
    private readonly Action<IntPtr>? _freeAction;
    private bool _disposed;

    public IntPtr HostPointer => _hostPtr;
    public IntPtr DevicePointer => _devPtr;
    public nuint SizeInBytes => _sizeInBytes;
    public int Length => _elementCount;
    public bool IsDisposed => _disposed;

    public Span<T> Span
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return new Span<T>((void*)_hostPtr, _elementCount);
        }
    }

    public ReadOnlySpan<T> ReadOnlySpan
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return new ReadOnlySpan<T>((void*)_hostPtr, _elementCount);
        }
    }

    public UnifiedMemoryBlock(IntPtr hostPtr, IntPtr devPtr, int elementCount, Action<IntPtr>? freeAction = null)
    {
        _hostPtr = hostPtr;
        _devPtr = devPtr;
        _elementCount = elementCount;
        _sizeInBytes = (nuint)(elementCount * sizeof(T));
        _freeAction = freeAction;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref T ItemRef(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)index >= (uint)_elementCount)
            throw new ArgumentOutOfRangeException(nameof(index));

        return ref ((T*)_hostPtr)[index];
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_hostPtr != IntPtr.Zero && _freeAction != null)
            {
                _freeAction(_hostPtr);
                _hostPtr = IntPtr.Zero;
                _devPtr = IntPtr.Zero;
            }
        }
    }
}
