using System;
using BenchmarkDotNet.Running;
using Glacier.Gpu.Benchmarks;

Console.WriteLine("Glacier.Gpu Benchmarks Runner");
Console.WriteLine("=============================");

if (args.Length > 0 && args[0].Equals("--bdn", StringComparison.OrdinalIgnoreCase))
{
    BenchmarkRunner.Run<DispatchBenchmarks>();
}
else
{
    Console.WriteLine("Running high-precision microbenchmarks...\n");
    using var bench = new DispatchBenchmarks();
    bench.Setup();

    Console.WriteLine("Warming up benchmarks...");
    for (int i = 0; i < 5; i++)
    {
        bench.RingBufferEnqueue();
    }

    double tradUs = bench.TraditionalLaunch();
    Console.WriteLine($"[Traditional Driver P/Invoke (cuLaunchKernel)]");
    Console.WriteLine($"  Average Dispatch Latency: {tradUs:F3} \u03bcs ({tradUs * 1000.0:F1} ns)");
    Console.WriteLine($"  Throughput: {1_000_000.0 / tradUs:N0} dispatches/sec\n");

    const int iterations = 100_000;
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        bench.RingBufferEnqueue();
    }
    sw.Stop();

    double avgUs = sw.Elapsed.TotalMicroseconds / iterations;
    Console.WriteLine($"[Persistent Ring Buffer Enqueue]");
    Console.WriteLine($"  Total Time: {sw.Elapsed.TotalMilliseconds:F2} ms for {iterations:N0} dispatches");
    Console.WriteLine($"  Average Dispatch Latency: {avgUs:F3} \u03bcs ({avgUs * 1000.0:F1} ns)");
    Console.WriteLine($"  Throughput: {1_000_000.0 / avgUs:N0} dispatches/sec");
    Console.WriteLine($"  Speedup: {tradUs / avgUs:F1}x faster than traditional driver dispatch\n");

    bench.Cleanup();
    Console.WriteLine("Benchmark run completed successfully.");
}
