using System;
using Glacier.Gpu.Common;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Xunit;

namespace Glacier.Gpu.Tests;

public unsafe class CudaGraphExecutorTests
{
    private static void LaunchVectorAdd(IntPtr fnVectorAdd, IntPtr dA, IntPtr dB, IntPtr dC, int count, IntPtr stream)
    {
        IntPtr a = dA;
        IntPtr b = dB;
        IntPtr c = dC;
        int n = count;
        void** pParams = stackalloc void*[4];
        pParams[0] = &a;
        pParams[1] = &b;
        pParams[2] = &c;
        pParams[3] = &n;

        uint gridX = (uint)((count + 255) / 256);
        CuDriver.Check(
            CuDriver.LaunchKernel(fnVectorAdd, gridX, 1, 1, 256, 1, 1, 0, stream, (IntPtr)pParams, IntPtr.Zero),
            "cuLaunchKernel(vector_add)");
    }

    [Fact]
    public void CudaGraphExecutor_MultiKernelCaptureReplayInvalidateAndDispose_SucceedsOnRealHardware()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");

        byte[] cubin = KernelCache.GetOrCompile(engine.DeviceInfo.Architecture, "VectorAdd", KernelSources.VectorAddPtx);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "cuModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAdd, module, "vector_add"), "cuModuleGetFunction");

        const int N = 2048;
        using var bufA = GpuBuffer<float>.AllocateCuda(N);
        using var bufB = GpuBuffer<float>.AllocateCuda(N);
        using var bufTmp = GpuBuffer<float>.AllocateCuda(N);
        using var bufOut = GpuBuffer<float>.AllocateCuda(N);

        try
        {
            float[] hostA = new float[N];
            float[] hostB = new float[N];
            for (int i = 0; i < N; i++)
            {
                hostA[i] = i * 1.0f;
                hostB[i] = 3.5f;
            }
            bufA.CopyFromHost(hostA);
            bufB.CopyFromHost(hostB);

            using var executor = new CudaGraphExecutor(engine);
            Assert.True(executor.OwnsStream);
            Assert.NotEqual(IntPtr.Zero, executor.StreamHandle);
            Assert.False(executor.IsCaptured);
            Assert.False(executor.IsFallbackActive);

            // Multi-kernel sequence: tmp = a + b, out = tmp + b  =>  out = a + 2*b
            void RecordTwoStageAdd(IntPtr stream)
            {
                LaunchVectorAdd(fnAdd, bufA.DevicePointer, bufB.DevicePointer, bufTmp.DevicePointer, N, stream);
                LaunchVectorAdd(fnAdd, bufTmp.DevicePointer, bufB.DevicePointer, bufOut.DevicePointer, N, stream);
            }

            // 1. First Execute captures, instantiates, and launches the graph
            executor.Execute(RecordTwoStageAdd, synchronize: true);
            Assert.True(executor.IsCaptured);
            Assert.False(executor.IsFallbackActive);
            Assert.NotEqual(IntPtr.Zero, executor.GraphExecHandle);
            Assert.Equal(1, executor.ReplayCount);
            Assert.Equal(0, executor.DirectExecutionCount);

            float[] result1 = new float[N];
            bufOut.CopyToHost(result1);
            for (int i = 0; i < N; i++)
            {
                Assert.Equal(hostA[i] + 2.0f * hostB[i], result1[i], precision: 4);
            }

            // 2. Update input VRAM in-place and Replay without re-capturing
            for (int i = 0; i < N; i++)
            {
                hostA[i] = (N - i) * 2.0f;
            }
            bufA.CopyFromHost(hostA, executor.StreamHandle);
            IntPtr capturedHandle = executor.GraphExecHandle;

            executor.Replay(synchronize: true);
            Assert.Equal(capturedHandle, executor.GraphExecHandle);
            Assert.Equal(2, executor.ReplayCount);

            float[] result2 = new float[N];
            bufOut.CopyToHost(result2);
            for (int i = 0; i < N; i++)
            {
                Assert.Equal(hostA[i] + 2.0f * hostB[i], result2[i], precision: 4);
            }

            // 3. Subsequent Execute reuses captured graph
            executor.Execute(RecordTwoStageAdd, synchronize: true);
            Assert.Equal(3, executor.ReplayCount);
            Assert.Equal(0, executor.DirectExecutionCount);

            // 4. Invalidate destroys executable graph and allows clean re-capture
            executor.Invalidate();
            Assert.False(executor.IsCaptured);
            Assert.Equal(IntPtr.Zero, executor.GraphExecHandle);

            Assert.True(executor.TryCapture(RecordTwoStageAdd));
            Assert.True(executor.IsCaptured);
            executor.Replay(synchronize: true);
            Assert.Equal(4, executor.ReplayCount);

            // 5. Disposal cleans up and blocks subsequent use
            executor.Dispose();
            Assert.True(executor.IsDisposed);
            Assert.False(executor.IsCaptured);
            Assert.Throws<ObjectDisposedException>(() => executor.Replay());
            Assert.Throws<ObjectDisposedException>(() => executor.Execute(RecordTwoStageAdd));
        }
        finally
        {
            CuDriver.CtxSetCurrent(engine.ContextHandle);
            CuDriver.ModuleUnload(module);
        }
    }

    [Fact]
    public void CudaGraphExecutor_MultiStreamForkJoin_CapturesAndReplaysCorrectly()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");

        byte[] cubin = KernelCache.GetOrCompile(engine.DeviceInfo.Architecture, "VectorAdd", KernelSources.VectorAddPtx);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "cuModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAdd, module, "vector_add"), "cuModuleGetFunction");

        CuDriver.Check(CuDriver.StreamCreate(out IntPtr auxStream, CuDriver.CU_STREAM_NON_BLOCKING), "cuStreamCreate(aux)");
        CuDriver.Check(CuDriver.EventCreate(out IntPtr forkEv, CuDriver.CU_EVENT_DISABLE_TIMING), "cuEventCreate(fork)");
        CuDriver.Check(CuDriver.EventCreate(out IntPtr joinEv, CuDriver.CU_EVENT_DISABLE_TIMING), "cuEventCreate(join)");

        const int N = 1024;
        using var bufA = GpuBuffer<float>.AllocateCuda(N);
        using var bufB = GpuBuffer<float>.AllocateCuda(N);
        using var bufD = GpuBuffer<float>.AllocateCuda(N);
        using var bufC1 = GpuBuffer<float>.AllocateCuda(N);
        using var bufC2 = GpuBuffer<float>.AllocateCuda(N);
        using var bufOut = GpuBuffer<float>.AllocateCuda(N);

        try
        {
            float[] a = new float[N];
            float[] b = new float[N];
            float[] d = new float[N];
            for (int i = 0; i < N; i++)
            {
                a[i] = i * 0.5f;
                b[i] = 10.0f;
                d[i] = 25.0f;
            }
            bufA.CopyFromHost(a);
            bufB.CopyFromHost(b);
            bufD.CopyFromHost(d);

            using var executor = new CudaGraphExecutor(engine);
            executor.Execute(primaryStream =>
            {
                executor.ForkStream(auxStream, forkEv);
                LaunchVectorAdd(fnAdd, bufA.DevicePointer, bufB.DevicePointer, bufC1.DevicePointer, N, primaryStream);
                LaunchVectorAdd(fnAdd, bufA.DevicePointer, bufD.DevicePointer, bufC2.DevicePointer, N, auxStream);
                executor.JoinStream(auxStream, joinEv);
                LaunchVectorAdd(fnAdd, bufC1.DevicePointer, bufC2.DevicePointer, bufOut.DevicePointer, N, primaryStream);
            }, synchronize: true);

            Assert.True(executor.IsCaptured);
            Assert.False(executor.IsFallbackActive);
            Assert.Equal(1, executor.ReplayCount);

            float[] output = new float[N];
            bufOut.CopyToHost(output);
            for (int i = 0; i < N; i++)
            {
                Assert.Equal(2.0f * a[i] + b[i] + d[i], output[i], precision: 4);
            }
        }
        finally
        {
            CuDriver.CtxSetCurrent(engine.ContextHandle);
            CuDriver.EventDestroy(forkEv);
            CuDriver.EventDestroy(joinEv);
            CuDriver.StreamDestroy(auxStream);
            CuDriver.ModuleUnload(module);
        }
    }

    [Fact]
    public void CudaGraphExecutor_UnsupportedCaptureOp_AutomaticallyFallsBackToDirectExecution()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");

        byte[] cubin = KernelCache.GetOrCompile(engine.DeviceInfo.Architecture, "VectorAdd", KernelSources.VectorAddPtx);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "cuModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAdd, module, "vector_add"), "cuModuleGetFunction");

        const int N = 512;
        using var bufA = GpuBuffer<float>.AllocateCuda(N);
        using var bufB = GpuBuffer<float>.AllocateCuda(N);
        using var bufC = GpuBuffer<float>.AllocateCuda(N);

        try
        {
            float[] a = new float[N];
            float[] b = new float[N];
            for (int i = 0; i < N; i++)
            {
                a[i] = i + 1.0f;
                b[i] = 100.0f;
            }
            bufA.CopyFromHost(a);
            bufB.CopyFromHost(b);

            using var executor = new CudaGraphExecutor(engine);

            // cuStreamSynchronize inside stream capture is forbidden by CUDA and forces capture failure;
            // CudaGraphExecutor must cleanly abort capture and fall back to direct execution.
            void SequenceWithStreamSync(IntPtr stream)
            {
                LaunchVectorAdd(fnAdd, bufA.DevicePointer, bufB.DevicePointer, bufC.DevicePointer, N, stream);
                CuDriver.Check(CuDriver.StreamSynchronize(stream), "cuStreamSynchronize(inside sequence)");
            }

            executor.Execute(SequenceWithStreamSync, synchronize: true);
            Assert.False(executor.IsCaptured);
            Assert.True(executor.IsFallbackActive);
            Assert.False(string.IsNullOrWhiteSpace(executor.LastCaptureError));
            Assert.Equal(0, executor.ReplayCount);
            Assert.Equal(1, executor.DirectExecutionCount);

            float[] c = new float[N];
            bufC.CopyToHost(c);
            for (int i = 0; i < N; i++)
            {
                Assert.Equal(a[i] + b[i], c[i], precision: 4);
            }

            // Second call continues using direct fallback cleanly
            executor.Execute(SequenceWithStreamSync, synchronize: true);
            Assert.Equal(2, executor.DirectExecutionCount);
        }
        finally
        {
            CuDriver.CtxSetCurrent(engine.ContextHandle);
            CuDriver.ModuleUnload(module);
        }
    }

    [Fact]
    public void CudaGraphExecutor_CallerOwnedStream_PreservesStreamAfterDispose()
    {
        if (!CuDriver.IsAvailable()) return;

        using var engine = new NvidiaSassEngine();
        engine.Initialize();
        CuDriver.Check(CuDriver.CtxSetCurrent(engine.ContextHandle), "cuCtxSetCurrent");
        CuDriver.Check(CuDriver.StreamCreate(out IntPtr callerStream, CuDriver.CU_STREAM_NON_BLOCKING), "cuStreamCreate");

        try
        {
            using (var executor = new CudaGraphExecutor(engine, callerStream))
            {
                Assert.False(executor.OwnsStream);
                Assert.Equal(callerStream, executor.StreamHandle);
                Assert.True(executor.TryCapture(_ => { }));
                executor.Replay(synchronize: true);
                Assert.Equal(1, executor.ReplayCount);
            }

            // Caller-owned stream must remain valid after executor disposal
            CuDriver.Check(CuDriver.StreamSynchronize(callerStream), "cuStreamSynchronize(after dispose)");
        }
        finally
        {
            CuDriver.CtxSetCurrent(engine.ContextHandle);
            CuDriver.StreamDestroy(callerStream);
        }
    }
}
