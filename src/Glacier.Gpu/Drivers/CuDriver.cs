using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Glacier.Gpu.Drivers;

/// <summary>
/// Direct P/Invoke bindings for the NVIDIA CUDA Driver API (nvcuda.dll / libcuda.so.1).
/// Bypasses cudart64.dll runtime overhead with Native AOT blittable signatures.
/// </summary>
public static unsafe class CuDriver
{
    private const string CudaLib = "nvcuda.dll";

    public const uint CU_MEMHOSTALLOC_PORTABLE = 0x01;
    public const uint CU_MEMHOSTALLOC_DEVICEMAP = 0x02;
    public const uint CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;
    public const uint CU_STREAM_DEFAULT = 0x00;
    public const uint CU_STREAM_NON_BLOCKING = 0x01;

    static CuDriver()
    {
        NativeDriverResolver.EnsureRegistered();
    }

    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            NativeDriverResolver.EnsureRegistered();
            bool loaded = false;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                loaded = NativeLibrary.TryLoad("nvcuda.dll", out IntPtr handle) && handle != IntPtr.Zero;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                loaded = (NativeLibrary.TryLoad("libcuda.so.1", out IntPtr handle) ||
                          NativeLibrary.TryLoad("libcuda.so", out handle)) && handle != IntPtr.Zero;
            }

            if (!loaded)
                return false;

            return Init(0) == 0 && DeviceGetCount(out int count) == 0 && count > 0;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    [DllImport(CudaLib, EntryPoint = "cuInit")]
    public static extern int Init(uint flags);

    [DllImport(CudaLib, EntryPoint = "cuDriverGetVersion")]
    public static extern int DriverGetVersion(out int driverVersion);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGetCount")]
    public static extern int DeviceGetCount(out int count);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGet")]
    public static extern int DeviceGet(out int device, int ordinal);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGetName")]
    public static extern int DeviceGetName(byte* name, int len, int dev);

    public static int DeviceGetName(byte[] name, int len, int dev)
    {
        ArgumentNullException.ThrowIfNull(name);
        int safeLen = Math.Min(len, name.Length);
        fixed (byte* pName = name)
        {
            return DeviceGetName(pName, safeLen, dev);
        }
    }

    [DllImport(CudaLib, EntryPoint = "cuDeviceTotalMem_v2")]
    public static extern int DeviceTotalMem(out nuint bytes, int dev);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGetAttribute")]
    public static extern int DeviceGetAttribute(out int pi, int attrib, int dev);

    [DllImport(CudaLib, EntryPoint = "cuCtxCreate_v2")]
    public static extern int CtxCreate(out IntPtr pctx, uint flags, int dev);

    [DllImport(CudaLib, EntryPoint = "cuCtxSetCurrent")]
    public static extern int CtxSetCurrent(IntPtr ctx);

    [DllImport(CudaLib, EntryPoint = "cuCtxGetCurrent")]
    public static extern int CtxGetCurrent(out IntPtr pctx);

    [DllImport(CudaLib, EntryPoint = "cuCtxDestroy_v2")]
    public static extern int CtxDestroy(IntPtr ctx);

    [DllImport(CudaLib, EntryPoint = "cuCtxSynchronize")]
    public static extern int CtxSynchronize();

    [DllImport(CudaLib, EntryPoint = "cuModuleLoadData")]
    public static extern int ModuleLoadData(out IntPtr module, void* image);

    public static int ModuleLoadData(out IntPtr module, byte[] image)
    {
        ArgumentNullException.ThrowIfNull(image);
        fixed (byte* pImage = image)
        {
            return ModuleLoadData(out module, pImage);
        }
    }

    [DllImport(CudaLib, EntryPoint = "cuModuleGetFunction")]
    public static extern int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, byte* name);

    public static int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        int byteCount = Encoding.UTF8.GetByteCount(name);
        byte[] utf8 = new byte[byteCount + 1];
        Encoding.UTF8.GetBytes(name, 0, name.Length, utf8, 0);
        fixed (byte* pName = utf8)
        {
            return ModuleGetFunction(out hfunc, hmod, pName);
        }
    }

    [DllImport(CudaLib, EntryPoint = "cuMemAlloc_v2")]
    public static extern int MemAlloc(out IntPtr dptr, nuint bytesize);

    [DllImport(CudaLib, EntryPoint = "cuMemFree_v2")]
    public static extern int MemFree(IntPtr dptr);

    [DllImport(CudaLib, EntryPoint = "cuMemGetInfo_v2")]
    public static extern int MemGetInfo(out nuint free, out nuint total);

    [DllImport(CudaLib, EntryPoint = "cuMemHostAlloc")]
    public static extern int MemHostAlloc(out IntPtr pp, nuint bytesize, uint flags);

    [DllImport(CudaLib, EntryPoint = "cuMemFreeHost")]
    public static extern int MemFreeHost(IntPtr p);

    [DllImport(CudaLib, EntryPoint = "cuMemHostGetDevicePointer_v2")]
    public static extern int MemHostGetDevicePointer(out IntPtr pdptr, IntPtr p, uint flags);

    [DllImport(CudaLib, EntryPoint = "cuMemcpyHtoD_v2")]
    public static extern int MemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, nuint byteCount);

    [DllImport(CudaLib, EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static extern int MemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, nuint byteCount, IntPtr hStream);

    [DllImport(CudaLib, EntryPoint = "cuMemcpyDtoH_v2")]
    public static extern int MemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, nuint byteCount);

    [DllImport(CudaLib, EntryPoint = "cuMemcpyDtoHAsync_v2")]
    public static extern int MemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, nuint byteCount, IntPtr hStream);

    [DllImport(CudaLib, EntryPoint = "cuMemcpyDtoD_v2")]
    public static extern int MemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount);

    [DllImport(CudaLib, EntryPoint = "cuMemcpyDtoDAsync_v2")]
    public static extern int MemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount, IntPtr hStream);

    [DllImport(CudaLib, EntryPoint = "cuLaunchKernel")]
    public static extern int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr kernelParams, IntPtr extra);

    [DllImport(CudaLib, EntryPoint = "cuStreamCreate")]
    public static extern int StreamCreate(out IntPtr phStream, uint flags);

    [DllImport(CudaLib, EntryPoint = "cuStreamSynchronize")]
    public static extern int StreamSynchronize(IntPtr hStream);

    [DllImport(CudaLib, EntryPoint = "cuStreamQuery")]
    public static extern int StreamQuery(IntPtr hStream);

    [DllImport(CudaLib, EntryPoint = "cuStreamDestroy_v2")]
    public static extern int StreamDestroy(IntPtr hStream);

    public static string GetDeviceName(int device)
    {
        byte* buf = stackalloc byte[256];
        NativeMemory.Clear(buf, 256);
        DeviceGetName(buf, 256, device);
        int len = 0;
        while (len < 256 && buf[len] != 0) len++;
        return Encoding.ASCII.GetString(buf, len);
    }

    public static string GetComputeCapability(int device)
    {
        DeviceGetAttribute(out int major, 75 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */, device);
        DeviceGetAttribute(out int minor, 76 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */, device);
        return $"sm_{major}{minor}";
    }

    public static int GetMultiprocessorCount(int device)
    {
        DeviceGetAttribute(out int smCount, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, device);
        return smCount;
    }

    public static int GetMaxThreadsPerBlock(int device)
    {
        DeviceGetAttribute(out int maxThreads, 1 /* CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK */, device);
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
