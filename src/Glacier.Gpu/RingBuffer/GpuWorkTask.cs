using System.Runtime.InteropServices;

namespace Glacier.Gpu.RingBuffer;

/// <summary>
/// 64-byte cache-line aligned task descriptor struct exchanged between CPU and GPU persistent worker.
/// Zero heap allocation on submission; lives in host-pinned device-mapped memory.
/// </summary>
[StructLayout(LayoutKind.Sequential, Size = 64)]
public struct GpuWorkTask
{
    public volatile uint TaskId;        // Monotonically increasing sequence ID
    public volatile uint OpCode;        // TaskOpCode
    public volatile uint ElementCount;  // N elements
    public volatile uint Status;        // 0 = Submitted/Running, 1 = Completed
    public ulong BufferA;               // GPU address of Buffer A
    public ulong BufferB;               // GPU address of Buffer B
    public ulong BufferC;               // GPU address of Buffer C
    public ulong BufferD;               // Optional 4th parameter or output
    public float Scalar;                // Scalar multiplier / weight
    public uint Flags;                  // Execution flags
    public uint Reserved0;
    public uint Reserved1;
}
