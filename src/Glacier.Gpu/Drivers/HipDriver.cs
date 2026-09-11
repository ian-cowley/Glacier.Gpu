using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Glacier.Gpu.Drivers;

/// <summary>
/// Direct P/Invoke bindings for the AMD HIP Driver API (amdhip64.dll).
/// Native Windows ROCm / HIP user-mode driver for Radeon GPUs and APUs.
/// </summary>
public static class HipDriver
{
    private const string HipLib = "amdhip64.dll";

    public const uint HIP_HOST_MALLOC_DEFAULT = 0x0;
    public const uint HIP_HOST_MALLOC_PORTABLE = 0x1;
    public const uint HIP_HOST_MALLOC_MAPPED = 0x2;
    public const uint HIP_HOST_MALLOC_WRITECOMBINED = 0x4;

    public const int HIP_MEMCPY_HOST_TO_HOST = 0;
    public const int HIP_MEMCPY_HOST_TO_DEVICE = 1;
    public const int HIP_MEMCPY_DEVICE_TO_HOST = 2;
    public const int HIP_MEMCPY_DEVICE_TO_DEVICE = 3;

    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            return NativeLibrary.TryLoad(HipLib, out IntPtr handle) && handle != IntPtr.Zero;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    [DllImport(HipLib, EntryPoint = "hipInit")]
    public static extern int Init(uint flags);

    [DllImport(HipLib, EntryPoint = "hipGetDeviceCount")]
    public static extern int GetDeviceCount(out int count);

    [DllImport(HipLib, EntryPoint = "hipDeviceGetName")]
    public static extern int DeviceGetName(byte[] name, int len, int dev);

    [DllImport(HipLib, EntryPoint = "hipSetDevice")]
    public static extern int SetDevice(int device);

    [DllImport(HipLib, EntryPoint = "hipDeviceSynchronize")]
    public static extern int DeviceSynchronize();

    [DllImport(HipLib, EntryPoint = "hipMalloc")]
    public static extern int Malloc(out IntPtr dptr, nuint size);

    [DllImport(HipLib, EntryPoint = "hipFree")]
    public static extern int Free(IntPtr dptr);

    [DllImport(HipLib, EntryPoint = "hipHostMalloc")]
    public static extern int HostMalloc(out IntPtr pptr, nuint size, uint flags);

    [DllImport(HipLib, EntryPoint = "hipHostFree")]
    public static extern int HostFree(IntPtr ptr);

    [DllImport(HipLib, EntryPoint = "hipHostGetDevicePointer")]
    public static extern int HostGetDevicePointer(out IntPtr devPtr, IntPtr hostPtr, uint flags);

    [DllImport(HipLib, EntryPoint = "hipMemcpy")]
    public static extern int Memcpy(IntPtr dst, IntPtr src, nuint size, int kind);

    [DllImport(HipLib, EntryPoint = "hipStreamCreate")]
    public static extern int StreamCreate(out IntPtr pStream);

    [DllImport(HipLib, EntryPoint = "hipStreamSynchronize")]
    public static extern int StreamSynchronize(IntPtr stream);

    [DllImport(HipLib, EntryPoint = "hipStreamDestroy")]
    public static extern int StreamDestroy(IntPtr stream);

    public static string GetDeviceName(int device)
    {
        var buf = new byte[256];
        DeviceGetName(buf, buf.Length, device);
        return Encoding.ASCII.GetString(buf).TrimEnd('\0');
    }

    public static void Check(int res, string op)
    {
        if (res != 0)
        {
            throw new InvalidOperationException($"AMD HIP Driver Error during '{op}': code {res}");
        }
    }
}
