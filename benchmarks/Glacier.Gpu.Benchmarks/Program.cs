using System;
using BenchmarkDotNet.Running;
using Glacier.Gpu.Benchmarks;

Console.WriteLine("Glacier.Gpu Benchmarks Runner");
Console.WriteLine("=============================");

if (args.Length > 0 && args[0].Equals("--bdn", StringComparison.OrdinalIgnoreCase))
{
    BenchmarkRunner.Run<DispatchBenchmarks>();
}
// Run Real-World Impact Benchmarks (Tokens/sec, VRAM Ceilings, Wave32 vs Wave64)
RealWorldBenchmarks.RunAll();

Console.WriteLine("All real-world benchmarks completed successfully.");
