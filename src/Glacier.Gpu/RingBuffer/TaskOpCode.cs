namespace Glacier.Gpu.RingBuffer;

/// <summary>
/// Micro-operation code executed by the GPU persistent worker megakernel.
/// </summary>
public enum TaskOpCode : uint
{
    Nop = 0,
    VectorAdd = 1,
    VectorFma = 2,
    MatrixTile = 3,
    ReduceSum = 4,
    CustomKernel = 10,
    Shutdown = 99
}
