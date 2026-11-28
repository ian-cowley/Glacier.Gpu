// <copyright file="CuDriver.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;

namespace Glacier.Gpu.Drivers;

#if USE_LOCAL_GLACIER_COMPUTE
using CanonicalCuDriver = global::Glacier.Compute.Drivers.CuDriver;

/// <summary>
/// Direct Native AOT-safe bindings for the NVIDIA CUDA Driver API (<c>nvcuda.dll</c> / <c>libcuda.so.1</c>).
/// Delegates directly to the canonical <see cref="global::Glacier.Compute.Drivers.CuDriver"/> implementation when building against local <c>Glacier.Compute</c>.
/// </summary>
public static unsafe partial class CuDriver
{
    public const uint CU_MEMHOSTALLOC_PORTABLE = CanonicalCuDriver.CU_MEMHOSTALLOC_PORTABLE;
    public const uint CU_MEMHOSTALLOC_DEVICEMAP = CanonicalCuDriver.CU_MEMHOSTALLOC_DEVICEMAP;
    public const uint CU_MEMHOSTALLOC_WRITECOMBINED = CanonicalCuDriver.CU_MEMHOSTALLOC_WRITECOMBINED;
    public const uint CU_STREAM_DEFAULT = CanonicalCuDriver.CU_STREAM_DEFAULT;
    public const uint CU_STREAM_NON_BLOCKING = CanonicalCuDriver.CU_STREAM_NON_BLOCKING;
    public const uint CU_EVENT_DEFAULT = CanonicalCuDriver.CU_EVENT_DEFAULT;
    public const uint CU_EVENT_DISABLE_TIMING = CanonicalCuDriver.CU_EVENT_DISABLE_TIMING;

    public const uint CU_STREAM_CAPTURE_MODE_GLOBAL = CanonicalCuDriver.CU_STREAM_CAPTURE_MODE_GLOBAL;
    public const uint CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = CanonicalCuDriver.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
    public const uint CU_STREAM_CAPTURE_MODE_RELAXED = CanonicalCuDriver.CU_STREAM_CAPTURE_MODE_RELAXED;

    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = CanonicalCuDriver.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = CanonicalCuDriver.CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;
    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = CanonicalCuDriver.CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = CanonicalCuDriver.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;

    public const int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = CanonicalCuDriver.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
    public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = CanonicalCuDriver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = CanonicalCuDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = CanonicalCuDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;

    public static bool IsAvailable() => CanonicalCuDriver.IsAvailable();

    public static int Init(uint flags) => CanonicalCuDriver.Init(flags);

    public static int DriverGetVersion(out int driverVersion) => CanonicalCuDriver.DriverGetVersion(out driverVersion);

    public static int DeviceGetCount(out int count) => CanonicalCuDriver.DeviceGetCount(out count);

    public static int DeviceGet(out int device, int ordinal) => CanonicalCuDriver.DeviceGet(out device, ordinal);

    public static int DeviceGetName(byte* name, int len, int dev) => CanonicalCuDriver.DeviceGetName(name, len, dev);

    public static int DeviceGetName(byte[] name, int len, int dev) => CanonicalCuDriver.DeviceGetName(name, len, dev);

    public static int DeviceTotalMem(out nuint bytes, int dev) => CanonicalCuDriver.DeviceTotalMem(out bytes, dev);

    public static int DeviceGetAttribute(out int pi, int attrib, int dev) => CanonicalCuDriver.DeviceGetAttribute(out pi, attrib, dev);

    public static int CtxCreate(out IntPtr pctx, uint flags, int dev) => CanonicalCuDriver.CtxCreate(out pctx, flags, dev);

    public static int CtxSetCurrent(IntPtr ctx) => CanonicalCuDriver.CtxSetCurrent(ctx);

    public static int CtxGetCurrent(out IntPtr pctx) => CanonicalCuDriver.CtxGetCurrent(out pctx);

    public static int CtxDestroy(IntPtr ctx) => CanonicalCuDriver.CtxDestroy(ctx);

    public static int CtxSynchronize() => CanonicalCuDriver.CtxSynchronize();

    public static int ModuleLoadData(out IntPtr module, void* image) => CanonicalCuDriver.ModuleLoadData(out module, image);

    public static int ModuleLoadData(out IntPtr module, byte[] image) => CanonicalCuDriver.ModuleLoadData(out module, image);

    public static int ModuleLoadData(out IntPtr module, ReadOnlySpan<byte> image) => CanonicalCuDriver.ModuleLoadData(out module, image);

    public static int ModuleLoadData(out IntPtr module, string ptxSource) => CanonicalCuDriver.ModuleLoadData(out module, ptxSource);

    public static int ModuleLoadDataEx(out IntPtr module, void* image, uint numOptions = 0, IntPtr options = default, IntPtr optionValues = default)
        => CanonicalCuDriver.ModuleLoadDataEx(out module, image, numOptions, options, optionValues);

    public static int ModuleLoadDataEx(out IntPtr module, string ptxSource) => CanonicalCuDriver.ModuleLoadDataEx(out module, ptxSource);

    public static int ModuleUnload(IntPtr hmod) => CanonicalCuDriver.ModuleUnload(hmod);

    public static int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, byte* name) => CanonicalCuDriver.ModuleGetFunction(out hfunc, hmod, name);

    public static int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name) => CanonicalCuDriver.ModuleGetFunction(out hfunc, hmod, name);

    public static int ModuleGetGlobal(out IntPtr dptr, out nuint bytes, IntPtr hmod, byte* name)
        => CanonicalCuDriver.ModuleGetGlobal(out dptr, out bytes, hmod, name);

    public static int ModuleGetGlobal(out IntPtr dptr, out nuint bytes, IntPtr hmod, string name)
        => CanonicalCuDriver.ModuleGetGlobal(out dptr, out bytes, hmod, name);

    public static int MemAlloc(out IntPtr dptr, nuint bytesize) => CanonicalCuDriver.MemAlloc(out dptr, bytesize);

    public static int MemFree(IntPtr dptr) => CanonicalCuDriver.MemFree(dptr);

    public static int MemGetInfo(out nuint free, out nuint total) => CanonicalCuDriver.MemGetInfo(out free, out total);

    public static int MemHostAlloc(out IntPtr pp, nuint bytesize, uint flags) => CanonicalCuDriver.MemHostAlloc(out pp, bytesize, flags);

    public static int MemFreeHost(IntPtr p) => CanonicalCuDriver.MemFreeHost(p);

    public static int MemHostRegister(IntPtr p, nuint bytesize, uint flags) => CanonicalCuDriver.MemHostRegister(p, bytesize, flags);

    public static int MemHostUnregister(IntPtr p) => CanonicalCuDriver.MemHostUnregister(p);

    public static int MemHostGetDevicePointer(out IntPtr pdptr, IntPtr p, uint flags)
        => CanonicalCuDriver.MemHostGetDevicePointer(out pdptr, p, flags);

    public static int MemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, nuint byteCount)
        => CanonicalCuDriver.MemcpyHtoD(dstDevice, srcHost, byteCount);

    public static int MemcpyHtoD(IntPtr dstDevice, void* srcHost, nuint byteCount)
        => CanonicalCuDriver.MemcpyHtoD(dstDevice, srcHost, byteCount);

    public static int MemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, nuint byteCount, IntPtr hStream)
        => CanonicalCuDriver.MemcpyHtoDAsync(dstDevice, srcHost, byteCount, hStream);

    public static int MemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, nuint byteCount)
        => CanonicalCuDriver.MemcpyDtoH(dstHost, srcDevice, byteCount);

    public static int MemcpyDtoH(void* dstHost, IntPtr srcDevice, nuint byteCount)
        => CanonicalCuDriver.MemcpyDtoH(dstHost, srcDevice, byteCount);

    public static int MemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, nuint byteCount, IntPtr hStream)
        => CanonicalCuDriver.MemcpyDtoHAsync(dstHost, srcDevice, byteCount, hStream);

    public static int MemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount)
        => CanonicalCuDriver.MemcpyDtoD(dstDevice, srcDevice, byteCount);

    public static int MemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount, IntPtr hStream)
        => CanonicalCuDriver.MemcpyDtoDAsync(dstDevice, srcDevice, byteCount, hStream);

    public static int MemsetD8(IntPtr dstDevice, byte uc, nuint count)
        => CanonicalCuDriver.MemsetD8(dstDevice, uc, count);

    public static int MemsetD8Async(IntPtr dstDevice, byte uc, nuint count, IntPtr hStream)
        => CanonicalCuDriver.MemsetD8Async(dstDevice, uc, count, hStream);

    public static int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr kernelParams, IntPtr extra)
        => CanonicalCuDriver.LaunchKernel(
            f, gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams, extra);

    public static int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr[] kernelParams)
        => CanonicalCuDriver.LaunchKernel(
            f, gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams);

    public static int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        ReadOnlySpan<IntPtr> kernelParams)
        => CanonicalCuDriver.LaunchKernel(
            f, gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams);

    public static int StreamCreate(out IntPtr phStream, uint flags) => CanonicalCuDriver.StreamCreate(out phStream, flags);

    public static int StreamSynchronize(IntPtr hStream) => CanonicalCuDriver.StreamSynchronize(hStream);

    public static int StreamQuery(IntPtr hStream) => CanonicalCuDriver.StreamQuery(hStream);

    public static int StreamDestroy(IntPtr hStream) => CanonicalCuDriver.StreamDestroy(hStream);

    public static int StreamBeginCapture(IntPtr hStream, uint mode) => CanonicalCuDriver.StreamBeginCapture(hStream, mode);

    public static int StreamEndCapture(IntPtr hStream, out IntPtr phGraph) => CanonicalCuDriver.StreamEndCapture(hStream, out phGraph);

    public static int EventCreate(out IntPtr phEvent, uint flags) => CanonicalCuDriver.EventCreate(out phEvent, flags);

    public static int EventRecord(IntPtr hEvent, IntPtr hStream) => CanonicalCuDriver.EventRecord(hEvent, hStream);

    public static int EventSynchronize(IntPtr hEvent) => CanonicalCuDriver.EventSynchronize(hEvent);

    public static int EventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd)
        => CanonicalCuDriver.EventElapsedTime(out pMilliseconds, hStart, hEnd);

    public static int EventDestroy(IntPtr hEvent) => CanonicalCuDriver.EventDestroy(hEvent);

    public static int StreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags = 0)
        => CanonicalCuDriver.StreamWaitEvent(hStream, hEvent, flags);

    public static int GraphInstantiateWithFlags(out IntPtr phGraphExec, IntPtr hGraph, ulong flags)
        => CanonicalCuDriver.GraphInstantiateWithFlags(out phGraphExec, hGraph, flags);

    public static int GraphLaunch(IntPtr hGraphExec, IntPtr hStream) => CanonicalCuDriver.GraphLaunch(hGraphExec, hStream);

    public static int GraphExecDestroy(IntPtr hGraphExec) => CanonicalCuDriver.GraphExecDestroy(hGraphExec);

    public static int GraphDestroy(IntPtr hGraph) => CanonicalCuDriver.GraphDestroy(hGraph);

    public static string GetDeviceName(int device) => CanonicalCuDriver.GetDeviceName(device);

    public static string GetComputeCapability(int device) => CanonicalCuDriver.GetComputeCapability(device);

    public static int GetMultiprocessorCount(int device) => CanonicalCuDriver.GetMultiprocessorCount(device);

    public static int GetMaxThreadsPerBlock(int device) => CanonicalCuDriver.GetMaxThreadsPerBlock(device);

    public static void Check(int res, string op) => CanonicalCuDriver.Check(res, op);
}
#else
/// <summary>
/// Standalone Native AOT-safe function-pointer bindings for the NVIDIA CUDA Driver API (<c>nvcuda.dll</c> / <c>libcuda.so.1</c>)
/// used when building against published NuGet <c>Glacier.Compute</c> packages without local <c>Glacier.Compute.Drivers.CuDriver</c>.
/// </summary>
public static unsafe partial class CuDriver
{
    public const uint CU_MEMHOSTALLOC_PORTABLE = 0x01;
    public const uint CU_MEMHOSTALLOC_DEVICEMAP = 0x02;
    public const uint CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;
    public const uint CU_STREAM_DEFAULT = 0x00;
    public const uint CU_STREAM_NON_BLOCKING = 0x01;
    public const uint CU_EVENT_DEFAULT = 0x00;
    public const uint CU_EVENT_DISABLE_TIMING = 0x02;

    public const uint CU_STREAM_CAPTURE_MODE_GLOBAL = 0;
    public const uint CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;
    public const uint CU_STREAM_CAPTURE_MODE_RELAXED = 2;

    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 0x01;
    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 0x02;
    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 0x04;
    public const ulong CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 0x08;

    public const int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    private static IntPtr s_driverHandle;

    private static delegate* unmanaged<uint, int> s_cuInit;
    private static delegate* unmanaged<int*, int> s_cuDriverGetVersion;
    private static delegate* unmanaged<int*, int> s_cuDeviceGetCount;
    private static delegate* unmanaged<int*, int, int> s_cuDeviceGet;
    private static delegate* unmanaged<byte*, int, int, int> s_cuDeviceGetName;
    private static delegate* unmanaged<nuint*, int, int> s_cuDeviceTotalMem;
    private static delegate* unmanaged<int*, int, int, int> s_cuDeviceGetAttribute;
    private static delegate* unmanaged<IntPtr*, uint, int, int> s_cuCtxCreate;
    private static delegate* unmanaged<IntPtr, int> s_cuCtxDestroy;
    private static delegate* unmanaged<int> s_cuCtxSynchronize;
    private static delegate* unmanaged<IntPtr*, void*, int> s_cuModuleLoadData;
    private static delegate* unmanaged<IntPtr*, void*, uint, IntPtr, IntPtr, int> s_cuModuleLoadDataEx;
    private static delegate* unmanaged<IntPtr, int> s_cuModuleUnload;
    private static delegate* unmanaged<IntPtr*, IntPtr, byte*, int> s_cuModuleGetFunction;
    private static delegate* unmanaged<IntPtr*, nuint*, IntPtr, byte*, int> s_cuModuleGetGlobal;
    private static delegate* unmanaged<IntPtr*, nuint, int> s_cuMemAlloc;
    private static delegate* unmanaged<IntPtr, int> s_cuMemFree;
    private static delegate* unmanaged<nuint*, nuint*, int> s_cuMemGetInfo;
    private static delegate* unmanaged<IntPtr*, nuint, uint, int> s_cuMemHostAlloc;
    private static delegate* unmanaged<IntPtr, int> s_cuMemFreeHost;
    private static delegate* unmanaged<IntPtr, nuint, uint, int> s_cuMemHostRegister;
    private static delegate* unmanaged<IntPtr, int> s_cuMemHostUnregister;
    private static delegate* unmanaged<IntPtr, IntPtr, nuint, int> s_cuMemcpyHtoD;
    private static delegate* unmanaged<IntPtr, IntPtr, nuint, IntPtr, int> s_cuMemcpyHtoDAsync;
    private static delegate* unmanaged<IntPtr, IntPtr, nuint, int> s_cuMemcpyDtoH;
    private static delegate* unmanaged<IntPtr, IntPtr, nuint, IntPtr, int> s_cuMemcpyDtoHAsync;
    private static delegate* unmanaged<IntPtr, IntPtr, nuint, int> s_cuMemcpyDtoD;
    private static delegate* unmanaged<IntPtr, byte, nuint, int> s_cuMemsetD8;
    private static delegate* unmanaged<IntPtr*, uint, int> s_cuStreamCreate;
    private static delegate* unmanaged<IntPtr, int> s_cuStreamSynchronize;
    private static delegate* unmanaged<IntPtr, int> s_cuStreamDestroy;
    private static delegate* unmanaged<IntPtr, uint, int> s_cuStreamBeginCapture;
    private static delegate* unmanaged<IntPtr, IntPtr*, int> s_cuStreamEndCapture;
    private static delegate* unmanaged<IntPtr*, uint, int> s_cuEventCreate;
    private static delegate* unmanaged<IntPtr, int> s_cuEventSynchronize;
    private static delegate* unmanaged<IntPtr, int> s_cuEventDestroy;
    private static delegate* unmanaged<IntPtr*, IntPtr, ulong, int> s_cuGraphInstantiateWithFlags;
    private static delegate* unmanaged<IntPtr, int> s_cuGraphExecDestroy;
    private static delegate* unmanaged<IntPtr, int> s_cuGraphDestroy;

    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, int> s_cuCtxSetCurrent;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr*, int> s_cuCtxGetCurrent;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr*, IntPtr, uint, int> s_cuMemHostGetDevicePointer;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, nuint, IntPtr, int> s_cuMemcpyDtoDAsync;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, byte, nuint, IntPtr, int> s_cuMemsetD8Async;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, uint, uint, uint, uint, uint, uint, uint, IntPtr, void**, void**, int> s_cuLaunchKernel;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, int> s_cuStreamQuery;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, int> s_cuEventRecord;
    private static delegate* unmanaged[SuppressGCTransition]<float*, IntPtr, IntPtr, int> s_cuEventElapsedTime;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, uint, int> s_cuStreamWaitEvent;
    private static delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, int> s_cuGraphLaunch;

    private static readonly Lazy<bool> s_isAvailable = new(() =>
    {
        try
        {
            if (s_driverHandle == IntPtr.Zero || s_cuInit == null || s_cuDeviceGetCount == null)
            {
                return false;
            }

            if (s_cuInit(0) != 0)
            {
                return false;
            }

            int count = 0;
            return s_cuDeviceGetCount(&count) == 0 && count > 0;
        }
        catch
        {
            return false;
        }
    });

    static CuDriver()
    {
        if (OperatingSystem.IsWindows())
        {
            NativeLibrary.TryLoad("nvcuda.dll", typeof(CuDriver).Assembly, DllImportSearchPath.System32, out s_driverHandle);
        }
        else if (OperatingSystem.IsLinux())
        {
            if (!NativeLibrary.TryLoad("libcuda.so.1", typeof(CuDriver).Assembly, null, out s_driverHandle))
            {
                NativeLibrary.TryLoad("libcuda.so", typeof(CuDriver).Assembly, null, out s_driverHandle);
            }
        }

        if (s_driverHandle == IntPtr.Zero)
        {
            return;
        }

        s_cuInit = (delegate* unmanaged<uint, int>)GetExport("cuInit");
        s_cuDriverGetVersion = (delegate* unmanaged<int*, int>)GetExport("cuDriverGetVersion");
        s_cuDeviceGetCount = (delegate* unmanaged<int*, int>)GetExport("cuDeviceGetCount");
        s_cuDeviceGet = (delegate* unmanaged<int*, int, int>)GetExport("cuDeviceGet");
        s_cuDeviceGetName = (delegate* unmanaged<byte*, int, int, int>)GetExport("cuDeviceGetName");
        s_cuDeviceTotalMem = (delegate* unmanaged<nuint*, int, int>)GetExport("cuDeviceTotalMem_v2", "cuDeviceTotalMem");
        s_cuDeviceGetAttribute = (delegate* unmanaged<int*, int, int, int>)GetExport("cuDeviceGetAttribute");
        s_cuCtxCreate = (delegate* unmanaged<IntPtr*, uint, int, int>)GetExport("cuCtxCreate_v2", "cuCtxCreate");
        s_cuCtxDestroy = (delegate* unmanaged<IntPtr, int>)GetExport("cuCtxDestroy_v2", "cuCtxDestroy");
        s_cuCtxSynchronize = (delegate* unmanaged<int>)GetExport("cuCtxSynchronize");
        s_cuModuleLoadData = (delegate* unmanaged<IntPtr*, void*, int>)GetExport("cuModuleLoadData");
        s_cuModuleLoadDataEx = (delegate* unmanaged<IntPtr*, void*, uint, IntPtr, IntPtr, int>)GetExport("cuModuleLoadDataEx");
        s_cuModuleUnload = (delegate* unmanaged<IntPtr, int>)GetExport("cuModuleUnload");
        s_cuModuleGetFunction = (delegate* unmanaged<IntPtr*, IntPtr, byte*, int>)GetExport("cuModuleGetFunction");
        s_cuModuleGetGlobal = (delegate* unmanaged<IntPtr*, nuint*, IntPtr, byte*, int>)GetExport("cuModuleGetGlobal_v2", "cuModuleGetGlobal");
        s_cuMemAlloc = (delegate* unmanaged<IntPtr*, nuint, int>)GetExport("cuMemAlloc_v2", "cuMemAlloc");
        s_cuMemFree = (delegate* unmanaged<IntPtr, int>)GetExport("cuMemFree_v2", "cuMemFree");
        s_cuMemGetInfo = (delegate* unmanaged<nuint*, nuint*, int>)GetExport("cuMemGetInfo_v2", "cuMemGetInfo");
        s_cuMemHostAlloc = (delegate* unmanaged<IntPtr*, nuint, uint, int>)GetExport("cuMemHostAlloc");
        s_cuMemFreeHost = (delegate* unmanaged<IntPtr, int>)GetExport("cuMemFreeHost");
        s_cuMemHostRegister = (delegate* unmanaged<IntPtr, nuint, uint, int>)GetExport("cuMemHostRegister_v2", "cuMemHostRegister");
        s_cuMemHostUnregister = (delegate* unmanaged<IntPtr, int>)GetExport("cuMemHostUnregister");
        s_cuMemcpyHtoD = (delegate* unmanaged<IntPtr, IntPtr, nuint, int>)GetExport("cuMemcpyHtoD_v2", "cuMemcpyHtoD");
        s_cuMemcpyHtoDAsync = (delegate* unmanaged<IntPtr, IntPtr, nuint, IntPtr, int>)GetExport("cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
        s_cuMemcpyDtoH = (delegate* unmanaged<IntPtr, IntPtr, nuint, int>)GetExport("cuMemcpyDtoH_v2", "cuMemcpyDtoH");
        s_cuMemcpyDtoHAsync = (delegate* unmanaged<IntPtr, IntPtr, nuint, IntPtr, int>)GetExport("cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
        s_cuMemcpyDtoD = (delegate* unmanaged<IntPtr, IntPtr, nuint, int>)GetExport("cuMemcpyDtoD_v2", "cuMemcpyDtoD");
        s_cuMemsetD8 = (delegate* unmanaged<IntPtr, byte, nuint, int>)GetExport("cuMemsetD8_v2", "cuMemsetD8");
        s_cuStreamCreate = (delegate* unmanaged<IntPtr*, uint, int>)GetExport("cuStreamCreate");
        s_cuStreamSynchronize = (delegate* unmanaged<IntPtr, int>)GetExport("cuStreamSynchronize");
        s_cuStreamDestroy = (delegate* unmanaged<IntPtr, int>)GetExport("cuStreamDestroy_v2", "cuStreamDestroy");
        s_cuStreamBeginCapture = (delegate* unmanaged<IntPtr, uint, int>)GetExport("cuStreamBeginCapture_v2", "cuStreamBeginCapture");
        s_cuStreamEndCapture = (delegate* unmanaged<IntPtr, IntPtr*, int>)GetExport("cuStreamEndCapture");
        s_cuEventCreate = (delegate* unmanaged<IntPtr*, uint, int>)GetExport("cuEventCreate");
        s_cuEventSynchronize = (delegate* unmanaged<IntPtr, int>)GetExport("cuEventSynchronize");
        s_cuEventDestroy = (delegate* unmanaged<IntPtr, int>)GetExport("cuEventDestroy_v2", "cuEventDestroy");
        s_cuGraphInstantiateWithFlags = (delegate* unmanaged<IntPtr*, IntPtr, ulong, int>)GetExport("cuGraphInstantiateWithFlags");
        s_cuGraphExecDestroy = (delegate* unmanaged<IntPtr, int>)GetExport("cuGraphExecDestroy");
        s_cuGraphDestroy = (delegate* unmanaged<IntPtr, int>)GetExport("cuGraphDestroy");

        s_cuCtxSetCurrent = (delegate* unmanaged[SuppressGCTransition]<IntPtr, int>)GetExport("cuCtxSetCurrent");
        s_cuCtxGetCurrent = (delegate* unmanaged[SuppressGCTransition]<IntPtr*, int>)GetExport("cuCtxGetCurrent");
        s_cuMemHostGetDevicePointer = (delegate* unmanaged[SuppressGCTransition]<IntPtr*, IntPtr, uint, int>)GetExport("cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer");
        s_cuMemcpyDtoDAsync = (delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, nuint, IntPtr, int>)GetExport("cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync");
        s_cuMemsetD8Async = (delegate* unmanaged[SuppressGCTransition]<IntPtr, byte, nuint, IntPtr, int>)GetExport("cuMemsetD8Async");
        s_cuLaunchKernel = (delegate* unmanaged[SuppressGCTransition]<IntPtr, uint, uint, uint, uint, uint, uint, uint, IntPtr, void**, void**, int>)GetExport("cuLaunchKernel");
        s_cuStreamQuery = (delegate* unmanaged[SuppressGCTransition]<IntPtr, int>)GetExport("cuStreamQuery");
        s_cuEventRecord = (delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, int>)GetExport("cuEventRecord");
        s_cuEventElapsedTime = (delegate* unmanaged[SuppressGCTransition]<float*, IntPtr, IntPtr, int>)GetExport("cuEventElapsedTime");
        s_cuStreamWaitEvent = (delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, uint, int>)GetExport("cuStreamWaitEvent");
        s_cuGraphLaunch = (delegate* unmanaged[SuppressGCTransition]<IntPtr, IntPtr, int>)GetExport("cuGraphLaunch");
    }

    private static IntPtr GetExport(string primaryName, string? fallbackName = null)
    {
        if (s_driverHandle != IntPtr.Zero)
        {
            if (NativeLibrary.TryGetExport(s_driverHandle, primaryName, out IntPtr addr))
            {
                return addr;
            }

            if (fallbackName is not null && NativeLibrary.TryGetExport(s_driverHandle, fallbackName, out addr))
            {
                return addr;
            }
        }

        return IntPtr.Zero;
    }

    private static void EnsureExport(void* fnPtr, string entryPoint)
    {
        if (s_driverHandle == IntPtr.Zero)
        {
            throw new PlatformNotSupportedException("NVIDIA CUDA Driver (nvcuda.dll / libcuda.so.1) is not available on this system.");
        }

        if (fnPtr == null)
        {
            throw new EntryPointNotFoundException($"CUDA driver entry point '{entryPoint}' was not exported by the loaded driver library.");
        }
    }

    public static bool IsAvailable() => s_isAvailable.Value;

    public static int Init(uint flags)
    {
        EnsureExport(s_cuInit, "cuInit");
        return s_cuInit(flags);
    }

    public static int DriverGetVersion(out int driverVersion)
    {
        EnsureExport(s_cuDriverGetVersion, "cuDriverGetVersion");
        int ver = 0;
        int res = s_cuDriverGetVersion(&ver);
        driverVersion = ver;
        return res;
    }

    public static int DeviceGetCount(out int count)
    {
        EnsureExport(s_cuDeviceGetCount, "cuDeviceGetCount");
        int c = 0;
        int res = s_cuDeviceGetCount(&c);
        count = c;
        return res;
    }

    public static int DeviceGet(out int device, int ordinal)
    {
        EnsureExport(s_cuDeviceGet, "cuDeviceGet");
        int dev = 0;
        int res = s_cuDeviceGet(&dev, ordinal);
        device = dev;
        return res;
    }

    public static int DeviceGetName(byte* name, int len, int dev)
    {
        EnsureExport(s_cuDeviceGetName, "cuDeviceGetName");
        return s_cuDeviceGetName(name, len, dev);
    }

    public static int DeviceGetName(byte[] name, int len, int dev)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentOutOfRangeException.ThrowIfNegative(len);
        EnsureExport(s_cuDeviceGetName, "cuDeviceGetName");
        int safeLen = Math.Min(len, name.Length);
        fixed (byte* pName = name)
        {
            return s_cuDeviceGetName(pName, safeLen, dev);
        }
    }

    public static int DeviceTotalMem(out nuint bytes, int dev)
    {
        EnsureExport(s_cuDeviceTotalMem, "cuDeviceTotalMem_v2");
        nuint total = 0;
        int res = s_cuDeviceTotalMem(&total, dev);
        bytes = total;
        return res;
    }

    public static int DeviceGetAttribute(out int pi, int attrib, int dev)
    {
        EnsureExport(s_cuDeviceGetAttribute, "cuDeviceGetAttribute");
        int val = 0;
        int res = s_cuDeviceGetAttribute(&val, attrib, dev);
        pi = val;
        return res;
    }

    public static int CtxCreate(out IntPtr pctx, uint flags, int dev)
    {
        EnsureExport(s_cuCtxCreate, "cuCtxCreate_v2");
        IntPtr ctx = IntPtr.Zero;
        int res = s_cuCtxCreate(&ctx, flags, dev);
        pctx = ctx;
        return res;
    }

    public static int CtxSetCurrent(IntPtr ctx)
    {
        EnsureExport(s_cuCtxSetCurrent, "cuCtxSetCurrent");
        return s_cuCtxSetCurrent(ctx);
    }

    public static int CtxGetCurrent(out IntPtr pctx)
    {
        EnsureExport(s_cuCtxGetCurrent, "cuCtxGetCurrent");
        IntPtr ctx = IntPtr.Zero;
        int res = s_cuCtxGetCurrent(&ctx);
        pctx = ctx;
        return res;
    }

    public static int CtxDestroy(IntPtr ctx)
    {
        EnsureExport(s_cuCtxDestroy, "cuCtxDestroy_v2");
        return s_cuCtxDestroy(ctx);
    }

    public static int CtxSynchronize()
    {
        EnsureExport(s_cuCtxSynchronize, "cuCtxSynchronize");
        return s_cuCtxSynchronize();
    }

    public static int ModuleLoadData(out IntPtr module, void* image)
    {
        EnsureExport(s_cuModuleLoadData, "cuModuleLoadData");
        IntPtr mod = IntPtr.Zero;
        int res = s_cuModuleLoadData(&mod, image);
        module = mod;
        return res;
    }

    public static int ModuleLoadData(out IntPtr module, byte[] image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return ModuleLoadData(out module, image.AsSpan());
    }

    public static int ModuleLoadData(out IntPtr module, ReadOnlySpan<byte> image)
    {
        EnsureExport(s_cuModuleLoadData, "cuModuleLoadData");
        IntPtr mod = IntPtr.Zero;
        fixed (byte* pImage = image)
        {
            int res = s_cuModuleLoadData(&mod, pImage);
            module = mod;
            return res;
        }
    }

    public static int ModuleLoadData(out IntPtr module, string ptxSource)
    {
        ArgumentNullException.ThrowIfNull(ptxSource);
        EnsureExport(s_cuModuleLoadData, "cuModuleLoadData");

        int utf8Len = Encoding.UTF8.GetByteCount(ptxSource);
        byte[]? rented = null;
        Span<byte> utf8 = utf8Len < 1024
            ? stackalloc byte[utf8Len + 1]
            : (rented = ArrayPool<byte>.Shared.Rent(utf8Len + 1));

        try
        {
            Encoding.UTF8.GetBytes(ptxSource, utf8[..utf8Len]);
            utf8[utf8Len] = 0;

            IntPtr mod = IntPtr.Zero;
            fixed (byte* pBytes = utf8)
            {
                int res = s_cuModuleLoadData(&mod, pBytes);
                module = mod;
                return res;
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    public static int ModuleLoadDataEx(out IntPtr module, void* image, uint numOptions = 0, IntPtr options = default, IntPtr optionValues = default)
    {
        if (s_driverHandle == IntPtr.Zero)
        {
            throw new PlatformNotSupportedException("NVIDIA CUDA Driver (nvcuda.dll / libcuda.so.1) is not available on this system.");
        }

        IntPtr mod = IntPtr.Zero;
        if (s_cuModuleLoadDataEx != null)
        {
            int res = s_cuModuleLoadDataEx(&mod, image, numOptions, options, optionValues);
            module = mod;
            return res;
        }

        if (s_cuModuleLoadData != null)
        {
            int res = s_cuModuleLoadData(&mod, image);
            module = mod;
            return res;
        }

        throw new EntryPointNotFoundException("Neither 'cuModuleLoadDataEx' nor 'cuModuleLoadData' was exported by the loaded CUDA driver.");
    }

    public static int ModuleLoadDataEx(out IntPtr module, string ptxSource)
    {
        ArgumentNullException.ThrowIfNull(ptxSource);
        if (s_driverHandle == IntPtr.Zero)
        {
            throw new PlatformNotSupportedException("NVIDIA CUDA Driver (nvcuda.dll / libcuda.so.1) is not available on this system.");
        }

        if (s_cuModuleLoadDataEx == null && s_cuModuleLoadData == null)
        {
            throw new EntryPointNotFoundException("Neither 'cuModuleLoadDataEx' nor 'cuModuleLoadData' was exported by the loaded CUDA driver.");
        }

        int utf8Len = Encoding.UTF8.GetByteCount(ptxSource);
        byte[]? rented = null;
        Span<byte> utf8 = utf8Len < 1024
            ? stackalloc byte[utf8Len + 1]
            : (rented = ArrayPool<byte>.Shared.Rent(utf8Len + 1));

        try
        {
            Encoding.UTF8.GetBytes(ptxSource, utf8[..utf8Len]);
            utf8[utf8Len] = 0;

            fixed (byte* pBytes = utf8)
            {
                return ModuleLoadDataEx(out module, pBytes);
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    public static int ModuleUnload(IntPtr hmod)
    {
        EnsureExport(s_cuModuleUnload, "cuModuleUnload");
        return s_cuModuleUnload(hmod);
    }

    public static int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, byte* name)
    {
        EnsureExport(s_cuModuleGetFunction, "cuModuleGetFunction");
        IntPtr fn = IntPtr.Zero;
        int res = s_cuModuleGetFunction(&fn, hmod, name);
        hfunc = fn;
        return res;
    }

    public static int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        EnsureExport(s_cuModuleGetFunction, "cuModuleGetFunction");

        int utf8Len = Encoding.UTF8.GetByteCount(name);
        byte[]? rented = null;
        Span<byte> utf8 = utf8Len < 256
            ? stackalloc byte[utf8Len + 1]
            : (rented = ArrayPool<byte>.Shared.Rent(utf8Len + 1));

        try
        {
            Encoding.UTF8.GetBytes(name, utf8[..utf8Len]);
            utf8[utf8Len] = 0;

            fixed (byte* pName = utf8)
            {
                return ModuleGetFunction(out hfunc, hmod, pName);
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    public static int ModuleGetGlobal(out IntPtr dptr, out nuint bytes, IntPtr hmod, byte* name)
    {
        EnsureExport(s_cuModuleGetGlobal, "cuModuleGetGlobal_v2");
        IntPtr ptr = IntPtr.Zero;
        nuint sz = 0;
        int res = s_cuModuleGetGlobal(&ptr, &sz, hmod, name);
        dptr = ptr;
        bytes = sz;
        return res;
    }

    public static int ModuleGetGlobal(out IntPtr dptr, out nuint bytes, IntPtr hmod, string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        EnsureExport(s_cuModuleGetGlobal, "cuModuleGetGlobal_v2");

        int utf8Len = Encoding.UTF8.GetByteCount(name);
        byte[]? rented = null;
        Span<byte> utf8 = utf8Len < 256
            ? stackalloc byte[utf8Len + 1]
            : (rented = ArrayPool<byte>.Shared.Rent(utf8Len + 1));

        try
        {
            Encoding.UTF8.GetBytes(name, utf8[..utf8Len]);
            utf8[utf8Len] = 0;

            fixed (byte* pName = utf8)
            {
                return ModuleGetGlobal(out dptr, out bytes, hmod, pName);
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    public static int MemAlloc(out IntPtr dptr, nuint bytesize)
    {
        EnsureExport(s_cuMemAlloc, "cuMemAlloc_v2");
        IntPtr ptr = IntPtr.Zero;
        int res = s_cuMemAlloc(&ptr, bytesize);
        dptr = ptr;
        return res;
    }

    public static int MemFree(IntPtr dptr)
    {
        EnsureExport(s_cuMemFree, "cuMemFree_v2");
        return s_cuMemFree(dptr);
    }

    public static int MemGetInfo(out nuint free, out nuint total)
    {
        EnsureExport(s_cuMemGetInfo, "cuMemGetInfo_v2");
        nuint f = 0;
        nuint t = 0;
        int res = s_cuMemGetInfo(&f, &t);
        free = f;
        total = t;
        return res;
    }

    public static int MemHostAlloc(out IntPtr pp, nuint bytesize, uint flags)
    {
        EnsureExport(s_cuMemHostAlloc, "cuMemHostAlloc");
        IntPtr p = IntPtr.Zero;
        int res = s_cuMemHostAlloc(&p, bytesize, flags);
        pp = p;
        return res;
    }

    public static int MemFreeHost(IntPtr p)
    {
        EnsureExport(s_cuMemFreeHost, "cuMemFreeHost");
        return s_cuMemFreeHost(p);
    }

    public static int MemHostRegister(IntPtr p, nuint bytesize, uint flags)
    {
        EnsureExport(s_cuMemHostRegister, "cuMemHostRegister_v2");
        return s_cuMemHostRegister(p, bytesize, flags);
    }

    public static int MemHostUnregister(IntPtr p)
    {
        EnsureExport(s_cuMemHostUnregister, "cuMemHostUnregister");
        return s_cuMemHostUnregister(p);
    }

    public static int MemHostGetDevicePointer(out IntPtr pdptr, IntPtr p, uint flags)
    {
        EnsureExport(s_cuMemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2");
        IntPtr dptr = IntPtr.Zero;
        int res = s_cuMemHostGetDevicePointer(&dptr, p, flags);
        pdptr = dptr;
        return res;
    }

    public static int MemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, nuint byteCount)
    {
        EnsureExport(s_cuMemcpyHtoD, "cuMemcpyHtoD_v2");
        return s_cuMemcpyHtoD(dstDevice, srcHost, byteCount);
    }

    public static int MemcpyHtoD(IntPtr dstDevice, void* srcHost, nuint byteCount)
        => MemcpyHtoD(dstDevice, (IntPtr)srcHost, byteCount);

    public static int MemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, nuint byteCount, IntPtr hStream)
    {
        EnsureExport(s_cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
        return s_cuMemcpyHtoDAsync(dstDevice, srcHost, byteCount, hStream);
    }

    public static int MemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, nuint byteCount)
    {
        EnsureExport(s_cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        return s_cuMemcpyDtoH(dstHost, srcDevice, byteCount);
    }

    public static int MemcpyDtoH(void* dstHost, IntPtr srcDevice, nuint byteCount)
        => MemcpyDtoH((IntPtr)dstHost, srcDevice, byteCount);

    public static int MemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, nuint byteCount, IntPtr hStream)
    {
        EnsureExport(s_cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
        return s_cuMemcpyDtoHAsync(dstHost, srcDevice, byteCount, hStream);
    }

    public static int MemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount)
    {
        EnsureExport(s_cuMemcpyDtoD, "cuMemcpyDtoD_v2");
        return s_cuMemcpyDtoD(dstDevice, srcDevice, byteCount);
    }

    public static int MemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount, IntPtr hStream)
    {
        EnsureExport(s_cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
        return s_cuMemcpyDtoDAsync(dstDevice, srcDevice, byteCount, hStream);
    }

    public static int MemsetD8(IntPtr dstDevice, byte uc, nuint count)
    {
        EnsureExport(s_cuMemsetD8, "cuMemsetD8_v2");
        return s_cuMemsetD8(dstDevice, uc, count);
    }

    public static int MemsetD8Async(IntPtr dstDevice, byte uc, nuint count, IntPtr hStream)
    {
        EnsureExport(s_cuMemsetD8Async, "cuMemsetD8Async");
        return s_cuMemsetD8Async(dstDevice, uc, count, hStream);
    }

    public static int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr kernelParams, IntPtr extra)
    {
        EnsureExport(s_cuLaunchKernel, "cuLaunchKernel");
        return s_cuLaunchKernel(
            f,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream,
            (void**)kernelParams, (void**)extra);
    }

    public static int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr[] kernelParams)
    {
        ArgumentNullException.ThrowIfNull(kernelParams);
        return LaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams.AsSpan());
    }

    public static int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        ReadOnlySpan<IntPtr> kernelParams)
    {
        EnsureExport(s_cuLaunchKernel, "cuLaunchKernel");

        if (kernelParams.IsEmpty)
        {
            return s_cuLaunchKernel(
                f,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream,
                null, null);
        }

        IntPtr[]? rented = null;
        Span<IntPtr> argPointers = kernelParams.Length <= 32
            ? stackalloc IntPtr[kernelParams.Length]
            : (rented = ArrayPool<IntPtr>.Shared.Rent(kernelParams.Length)).AsSpan(0, kernelParams.Length);

        try
        {
            fixed (IntPtr* pParams = kernelParams)
            fixed (IntPtr* pArgPointers = argPointers)
            {
                for (int i = 0; i < kernelParams.Length; i++)
                {
                    pArgPointers[i] = (IntPtr)(&pParams[i]);
                }

                return s_cuLaunchKernel(
                    f,
                    gridDimX, gridDimY, gridDimZ,
                    blockDimX, blockDimY, blockDimZ,
                    sharedMemBytes, hStream,
                    (void**)pArgPointers, null);
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<IntPtr>.Shared.Return(rented);
            }
        }
    }

    public static int StreamCreate(out IntPtr phStream, uint flags)
    {
        EnsureExport(s_cuStreamCreate, "cuStreamCreate");
        IntPtr stream = IntPtr.Zero;
        int res = s_cuStreamCreate(&stream, flags);
        phStream = stream;
        return res;
    }

    public static int StreamSynchronize(IntPtr hStream)
    {
        EnsureExport(s_cuStreamSynchronize, "cuStreamSynchronize");
        return s_cuStreamSynchronize(hStream);
    }

    public static int StreamQuery(IntPtr hStream)
    {
        EnsureExport(s_cuStreamQuery, "cuStreamQuery");
        return s_cuStreamQuery(hStream);
    }

    public static int StreamDestroy(IntPtr hStream)
    {
        EnsureExport(s_cuStreamDestroy, "cuStreamDestroy_v2");
        return s_cuStreamDestroy(hStream);
    }

    public static int StreamBeginCapture(IntPtr hStream, uint mode)
    {
        EnsureExport(s_cuStreamBeginCapture, "cuStreamBeginCapture_v2");
        return s_cuStreamBeginCapture(hStream, mode);
    }

    public static int StreamEndCapture(IntPtr hStream, out IntPtr phGraph)
    {
        EnsureExport(s_cuStreamEndCapture, "cuStreamEndCapture");
        IntPtr graph = IntPtr.Zero;
        int res = s_cuStreamEndCapture(hStream, &graph);
        phGraph = graph;
        return res;
    }

    public static int EventCreate(out IntPtr phEvent, uint flags)
    {
        EnsureExport(s_cuEventCreate, "cuEventCreate");
        IntPtr ev = IntPtr.Zero;
        int res = s_cuEventCreate(&ev, flags);
        phEvent = ev;
        return res;
    }

    public static int EventRecord(IntPtr hEvent, IntPtr hStream)
    {
        EnsureExport(s_cuEventRecord, "cuEventRecord");
        return s_cuEventRecord(hEvent, hStream);
    }

    public static int EventSynchronize(IntPtr hEvent)
    {
        EnsureExport(s_cuEventSynchronize, "cuEventSynchronize");
        return s_cuEventSynchronize(hEvent);
    }

    public static int EventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd)
    {
        EnsureExport(s_cuEventElapsedTime, "cuEventElapsedTime");
        float ms = 0f;
        int res = s_cuEventElapsedTime(&ms, hStart, hEnd);
        pMilliseconds = ms;
        return res;
    }

    public static int EventDestroy(IntPtr hEvent)
    {
        EnsureExport(s_cuEventDestroy, "cuEventDestroy_v2");
        return s_cuEventDestroy(hEvent);
    }

    public static int StreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags = 0)
    {
        EnsureExport(s_cuStreamWaitEvent, "cuStreamWaitEvent");
        return s_cuStreamWaitEvent(hStream, hEvent, flags);
    }

    public static int GraphInstantiateWithFlags(out IntPtr phGraphExec, IntPtr hGraph, ulong flags)
    {
        EnsureExport(s_cuGraphInstantiateWithFlags, "cuGraphInstantiateWithFlags");
        IntPtr graphExec = IntPtr.Zero;
        int res = s_cuGraphInstantiateWithFlags(&graphExec, hGraph, flags);
        phGraphExec = graphExec;
        return res;
    }

    public static int GraphLaunch(IntPtr hGraphExec, IntPtr hStream)
    {
        EnsureExport(s_cuGraphLaunch, "cuGraphLaunch");
        return s_cuGraphLaunch(hGraphExec, hStream);
    }

    public static int GraphExecDestroy(IntPtr hGraphExec)
    {
        EnsureExport(s_cuGraphExecDestroy, "cuGraphExecDestroy");
        return s_cuGraphExecDestroy(hGraphExec);
    }

    public static int GraphDestroy(IntPtr hGraph)
    {
        EnsureExport(s_cuGraphDestroy, "cuGraphDestroy");
        return s_cuGraphDestroy(hGraph);
    }

    public static string GetDeviceName(int device)
    {
        byte* buf = stackalloc byte[256];
        NativeMemory.Clear(buf, 256);
        int res = DeviceGetName(buf, 256, device);
        Check(res, nameof(DeviceGetName));
        int len = 0;
        while (len < 256 && buf[len] != 0)
        {
            len++;
        }

        return Encoding.ASCII.GetString(buf, len);
    }

    public static string GetComputeCapability(int device)
    {
        Check(DeviceGetAttribute(out int major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), nameof(GetComputeCapability));
        Check(DeviceGetAttribute(out int minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), nameof(GetComputeCapability));
        return $"sm_{major}{minor}";
    }

    public static int GetMultiprocessorCount(int device)
    {
        Check(DeviceGetAttribute(out int smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), nameof(GetMultiprocessorCount));
        return smCount;
    }

    public static int GetMaxThreadsPerBlock(int device)
    {
        Check(DeviceGetAttribute(out int maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device), nameof(GetMaxThreadsPerBlock));
        return maxThreads > 0 ? maxThreads : 1024;
    }

    public static void Check(int res, string op)
    {
        if (res != 0)
        {
            throw new InvalidOperationException($"CUDA Driver Error during '{op}': code {res}");
        }
    }
}
#endif
