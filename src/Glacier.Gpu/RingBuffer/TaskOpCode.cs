using System;

namespace Glacier.Gpu.RingBuffer;

/// <summary>
/// Micro-operation code executed by the GPU persistent worker megakernel.
/// </summary>
public enum TaskOpCode : uint
{
    Nop = 0,
    VectorAdd = 1,
    VectorFma = 2,

    [Obsolete("MatrixTile is not implemented by PersistentWorkerPtx; submitting this opcode throws NotSupportedException.")]
    MatrixTile = 3,

    [Obsolete("ReduceSum is not implemented by PersistentWorkerPtx; submitting this opcode throws NotSupportedException.")]
    ReduceSum = 4,

    [Obsolete("CustomKernel is not implemented by PersistentWorkerPtx; submitting this opcode throws NotSupportedException.")]
    CustomKernel = 10,

    Shutdown = 99
}
