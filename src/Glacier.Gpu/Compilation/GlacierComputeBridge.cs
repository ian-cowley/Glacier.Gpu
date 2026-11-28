// <copyright file="GlacierComputeBridge.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Glacier.Gpu.Compilation;

using System;
using System.Runtime.InteropServices;
using Glacier.Compute;
using Glacier.Compute.Assembler;
using Glacier.Compute.Assembler.Dxbc;
using Glacier.Compute.Assembler.Ptx;
using Glacier.Compute.Assembler.SpirV;
using Glacier.Compute.Memory;

/// <summary>
/// First-party compute engine bridge connecting <c>Glacier.Gpu</c> with <c>Glacier.Compute</c>.
/// Exposes in-memory PTX 8.0 emission, SPIR-V 1.6 assembly &amp; validation, precompiled DXBC/DXIL packaging,
/// and high-performance slab memory pooling.
/// <para>
/// Note: Direct LLVM 3.7 DXIL synthesis from <see cref="Instruction"/> streams (<see cref="ShaderBytecodeBuilder.BuildDxil"/>)
/// is unsupported without external DXC compilation and <c>dxil.dll</c> signing and throws <see cref="NotSupportedException"/>.
/// </para>
/// </summary>
public static class GlacierComputeBridge
{
    /// <summary>
    /// Creates a PTX kernel builder configured for the Glacier heterogeneous GPU runtime.
    /// </summary>
    public static PtxKernelBuilder CreatePtxBuilder(string kernelName, int targetArchSm = 89)
    {
        return new PtxKernelBuilder(kernelName).SetTargetArch($"sm_{targetArchSm}");
    }

    /// <summary>
    /// Creates a unified <see cref="ShaderBytecodeBuilder"/> for PTX 8.0 and SPIR-V 1.6 synthesis.
    /// </summary>
    public static ShaderBytecodeBuilder CreateBytecodeBuilder()
    {
        return new ShaderBytecodeBuilder();
    }

    /// <summary>
    /// Creates a pure-C# SPIR-V 1.6 binary module emitter.
    /// </summary>
    public static SpirvBinaryEmitter CreateSpirvEmitter()
    {
        return new SpirvBinaryEmitter();
    }

    /// <summary>
    /// Emits a binary SPIR-V 1.6 compute shader module from intermediate <see cref="Instruction"/>s.
    /// </summary>
    public static byte[] CompileSpirV(
        ReadOnlySpan<Instruction> instructions,
        ShaderStage stage = ShaderStage.Compute,
        uint localSizeX = 64,
        uint localSizeY = 1,
        uint localSizeZ = 1)
    {
        var emitter = new SpirvBinaryEmitter();
        return emitter.EmitModule(stage, instructions, localSizeX, localSizeY, localSizeZ).ToArray();
    }

    /// <summary>
    /// Emits a binary SPIR-V 1.6 compute shader module as 32-bit SPIR-V words ready for <c>VulkanEngine.DispatchCompute</c>.
    /// </summary>
    public static uint[] CompileSpirVWords(
        ReadOnlySpan<Instruction> instructions,
        ShaderStage stage = ShaderStage.Compute,
        uint localSizeX = 64,
        uint localSizeY = 1,
        uint localSizeZ = 1)
    {
        byte[] bytes = CompileSpirV(instructions, stage, localSizeX, localSizeY, localSizeZ);
        uint[] words = new uint[bytes.Length / sizeof(uint)];
        MemoryMarshal.Cast<byte, uint>(bytes).CopyTo(words);
        return words;
    }

    /// <summary>
    /// Validates a binary SPIR-V module header, instruction framing, and ID bounds.
    /// </summary>
    public static bool ValidateSpirV(ReadOnlySpan<byte> spirvBytes, out string? errorMessage)
    {
        return SpirvValidator.Validate(spirvBytes, out errorMessage);
    }

    /// <summary>
    /// Emits NVIDIA PTX 8.0 assembly text from intermediate <see cref="Instruction"/>s.
    /// </summary>
    public static string CompilePtx(
        ReadOnlySpan<Instruction> instructions,
        string kernelName = "glacier_compute_kernel",
        string targetArch = PtxEmitter.DefaultTargetArch)
    {
        return PtxEmitter.Emit(instructions, kernelName, targetArch);
    }

    /// <summary>
    /// Packages a precompiled LLVM 3.7 DXIL bitcode payload and PSV0 metadata chunk into a DXBC container.
    /// </summary>
    public static byte[] PackagePrecompiledDxbc(
        ReadOnlySpan<byte> precompiledBitcode,
        ShaderStage stage = ShaderStage.Compute,
        uint threadsX = 64,
        uint threadsY = 1,
        uint threadsZ = 1)
    {
        return DxbcContainerBuilder.PackagePrecompiledShader(stage, precompiledBitcode, threadsX, threadsY, threadsZ).ToArray();
    }

    /// <summary>
    /// Creates a high-performance host-backed slab allocator with lazy arena initialization.
    /// </summary>
    public static SlabAllocator CreateSlabAllocator(int slotsPerBin = 1024)
    {
        return new SlabAllocator(slotsPerBin);
    }

    /// <summary>
    /// Creates a high-performance device-resident or pinned-memory slab allocator backed by an external arena
    /// (for example <c>cuMemAlloc</c>, <c>cuMemHostAlloc</c>, or Direct3D 12 zero-copy heaps).
    /// </summary>
    public static SlabAllocator CreateSlabAllocator(IntPtr baseAddress, ulong byteLength, Action<IntPtr>? releaseAction = null)
    {
        return new SlabAllocator(baseAddress, byteLength, releaseAction);
    }
}
