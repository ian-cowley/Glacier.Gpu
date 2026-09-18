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

    static HipDriver()
    {
        NativeDriverResolver.EnsureRegistered();
    }

    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            NativeDriverResolver.EnsureRegistered();
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return (NativeLibrary.TryLoad("amdhip64.dll", out IntPtr handle) ||
                        NativeLibrary.TryLoad("amdhip64_6.dll", out handle) ||
                        NativeLibrary.TryLoad("amdhip64_7.dll", out handle)) && handle != IntPtr.Zero;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return (NativeLibrary.TryLoad("libamdhip64.so", out IntPtr handle) ||
                        NativeLibrary.TryLoad("libamdhip64.so.7", out handle) ||
                        NativeLibrary.TryLoad("libamdhip64.so.6", out handle) ||
                        NativeLibrary.TryLoad("/opt/rocm/lib/libamdhip64.so", out handle) ||
                        NativeLibrary.TryLoad("/opt/rocm-7.2.0/lib/libamdhip64.so", out handle)) && handle != IntPtr.Zero;
            }
            return false;
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

    [DllImport(HipLib, EntryPoint = "hipDeviceGetAttribute")]
    public static extern int DeviceGetAttribute(out int pi, int attr, int device);

    [DllImport(HipLib, EntryPoint = "hipDeviceTotalMem")]
    public static extern int DeviceTotalMem(out nuint bytes, int device);

    public const int HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    public const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    public static string GetDeviceName(int device)
    {
        var buf = new byte[256];
        DeviceGetName(buf, buf.Length, device);
        return Encoding.ASCII.GetString(buf).TrimEnd('\0');
    }

    public static int GetMultiprocessorCount(int device)
    {
        try
        {
            if (DeviceGetAttribute(out int count, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == 0 && count > 0)
                return count;
        }
        catch { }

        // Fallback based on known device name
        string name = GetDeviceName(device).ToLowerInvariant();
        if (name.Contains("890m")) return 16;
        if (name.Contains("880m") || name.Contains("780m") || name.Contains("680m")) return 12;
        if (name.Contains("760m")) return 8;
        if (name.Contains("660m")) return 6;
        if (name.Contains("7900 xtx")) return 96;
        if (name.Contains("7900 xt")) return 84;
        if (name.Contains("7800")) return 60;
        if (name.Contains("7700")) return 54;
        if (name.Contains("7600")) return 32;
        if (name.Contains("6900") || name.Contains("6800")) return 72;
        return 12;
    }

    public static nuint GetTotalMemory(int device)
    {
        try
        {
            if (DeviceTotalMem(out nuint mem, device) == 0 && mem > 0)
                return mem;
        }
        catch { }
        return unchecked((nuint)(12L * 1024 * 1024 * 1024)); // Default 12 GB
    }

    public static void Check(int res, string op)
    {
        if (res != 0)
        {
            throw new InvalidOperationException($"AMD HIP Driver Error during '{op}': code {res}");
        }
    }
}
