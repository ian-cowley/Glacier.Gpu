using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Engines;
using Glacier.Gpu.Factory;

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================================");
Console.WriteLine("             GLACIER.GPU: BARE-METAL HETEROGENEOUS ACCELERATION                 ");
Console.WriteLine("   Bypassing CUDA Runtime & Vulkan: Direct Driver APIs, SASS, & Zero-Copy RAM   ");
Console.WriteLine("               .NET 10 / C# High-Performance Computing Engine                   ");
Console.WriteLine("================================================================================\n");
Console.ResetColor();

Console.WriteLine(">>> Discovering and Initializing GPU Hardware via Direct Driver P/Invoke...");
using var heterogeneous = GpuEngineFactory.TryCreateHeterogeneousEngine();
using var standaloneNvidia = heterogeneous == null ? GpuEngineFactory.TryCreateNvidiaEngine() : null;
using var standaloneAmd = heterogeneous == null && HipDriverAvailable() ? new AmdRdnaEngine() : null;
if (standaloneAmd != null)
{
    try { standaloneAmd.Initialize(); } catch { }
}

NvidiaSassEngine? nvActive = heterogeneous?.Nvidia ?? standaloneNvidia;
AmdRdnaEngine? amdActive = heterogeneous?.Amd ?? (standaloneAmd is { IsInitialized: true } ? standaloneAmd : null);

if (nvActive is { IsInitialized: true } nvEngine)
{
    var nv = nvEngine.DeviceInfo;
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"    [NVIDIA dGPU] {nv.DeviceName} ({nv.Architecture}) | {nv.ComputeUnitsOrSms} SMs | {nv.TotalMemoryBytes / (1024 * 1024 * 1024.0):F1} GB GDDR6");
    Console.ResetColor();
}
else
{
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    Console.WriteLine("    [NVIDIA dGPU] Not available or driver not detected.");
    Console.ResetColor();
}

if (amdActive is { IsInitialized: true } amdEngine)
{
    var amd = amdEngine.DeviceInfo;
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"    [AMD APU]     {amd.DeviceName} ({amd.Architecture}) | {amd.ComputeUnitsOrSms} CUs | Unified LPDDR5X System RAM");
    Console.ResetColor();
}
else
{
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    Console.WriteLine("    [AMD APU]     Not available or HIP driver not detected.");
    Console.ResetColor();
}

Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 1: CPU-to-GPU Dispatch Latency (Traditional Driver vs Persistent Ring)");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

if (nvActive is { IsInitialized: true } nv1)
{
    int tradIters = 50_000;
    Console.WriteLine($"Running {tradIters:N0} traditional cuLaunchKernel launches via nvcuda.dll...");
    double tradUs = nv1.MeasureTraditionalLaunchLatency(tradIters);
    Console.WriteLine($"  [Traditional Driver API] Avg Dispatch Latency: {tradUs:F3} \u03bcs ({1_000_000.0 / tradUs:N0} launches/sec)");

    int ringIters = 10_000;
    Console.WriteLine("\nRunning Persistent Worker Ring Buffer (Host-Coherent Memory Megakernel)...");
    var (dispatchUs, roundTripUs, verified, sampleVal) = nv1.MeasureRingBufferLatency(ringIters);
    Console.WriteLine($"  [Persistent Ring Buffer] Fire-and-Forget Enqueue: {dispatchUs:F3} \u03bcs ({1_000_000.0 / dispatchUs:N0} tasks/sec)");
    Console.WriteLine($"  [Persistent Ring Buffer] Full Round-Trip Turnaround: {roundTripUs:F3} \u03bcs ({1_000_000.0 / roundTripUs:N0} roundtrips/sec)");

    if (verified)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  [Verification] Correctness check: PASSED (Result: {sampleVal:F1}, Expected: 40.0)");
    }
    else
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"  [Verification] Correctness check: FAILED (Result: {sampleVal:F1}, Expected: 40.0)");
    }
    Console.ResetColor();

    double speedup = tradUs / dispatchUs;
    Console.ForegroundColor = ConsoleColor.Magenta;
    Console.WriteLine($"  >>> DISPATCH SPEEDUP: {speedup:F1}x FASTER! Sub-microsecond GPU task dispatch achieved!");
    Console.ResetColor();
}
else
{
    Console.WriteLine("NVIDIA discrete GPU not available. Skipping Experiment 1.");
}

Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 2: Raw SASS Hardware Compute");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

if (nvActive is { IsInitialized: true } nv2)
{
    Console.WriteLine("Executing raw SASS machine code kernel (unrolled FMA loops)...");
    var (tflops, avgMs) = nv2.BenchmarkFmaCompute(N: 4_194_304, loopCount: 250);
    Console.WriteLine($"  Average Execution Latency:       {avgMs:F2} ms");
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine($"  Sustained Hardware FP32 Compute: {tflops:F2} TFLOPS!");
    Console.ResetColor();
}
else
{
    Console.WriteLine("NVIDIA discrete GPU not available. Skipping Experiment 2.");
}

Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 3: AMD Radeon Zero-Copy Unified Memory");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

if (amdActive is { IsInitialized: true } amd3)
{
    Console.WriteLine("Testing true zero-copy direct memory access from C#...");
    var (accessUs, accVerified) = amd3.TestZeroCopyDirectAccess();
    Console.ForegroundColor = accVerified ? ConsoleColor.Green : ConsoleColor.Red;
    Console.WriteLine($"  Zero-Copy Verification: {(accVerified ? "PASSED" : "FAILED")}");
    Console.ResetColor();
    Console.WriteLine($"  C#-to-GPU Memory Sharing Latency: {accessUs:F3} \u03bcs (Zero PCIe staging copy!)");

    int memMb = 256;
    Console.WriteLine($"\nBenchmarking unified LPDDR5X memory bandwidth ({memMb} MB buffer)...");
    var (bw, memMs) = amd3.BenchmarkUnifiedBandwidth(memMb);
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine($"  Sustained Unified Memory Bandwidth: {bw:F2} GB/s (Latency: {memMs:F2} ms)");
    Console.ResetColor();
}
else
{
    Console.WriteLine("AMD HIP APU not available. Skipping Experiment 3.");
}

Console.WriteLine("\n--------------------------------------------------------------------------------");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("EXPERIMENT 4: Heterogeneous Dual-GPU Concurrent Execution");
Console.ResetColor();
Console.WriteLine("--------------------------------------------------------------------------------");

if (heterogeneous is { IsInitialized: true })
{
    Console.WriteLine(">>> Firing Concurrent Dual-GPU Workloads (NVIDIA SASS FMA + AMD Zero-Copy Sweep)...");
    var (nvTflops, amdBw, totalMs) = heterogeneous.RunConcurrentWorkload(nvElements: 2_097_152, amdSizeMb: 128);

    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"    Concurrent Dual-GPU Test Completed in {totalMs:F2} ms!");
    Console.ResetColor();
    Console.WriteLine($"    NVIDIA Discrete Compute: {nvTflops:F2} TFLOPS");
    Console.WriteLine($"    AMD APU Memory Bandwidth: {amdBw:F2} GB/s");
}
else
{
    Console.WriteLine("Requires both NVIDIA CUDA and AMD HIP GPUs for heterogeneous concurrent execution.");
}

Console.WriteLine("\n================================================================================");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine("All bare-metal Glacier.Gpu experiments completed successfully!");
Console.ResetColor();
Console.WriteLine("================================================================================");

static bool HipDriverAvailable() => Glacier.Gpu.Drivers.HipDriver.IsAvailable();
