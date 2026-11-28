// <copyright file="HipDriver.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Glacier.Gpu.Drivers;

/// <summary>
/// Direct P/Invoke bindings for the AMD HIP Driver API (amdhip64.dll / libamdhip64.so).
/// Native ROCm / HIP user-mode driver for Radeon GPUs and APUs with blittable Native AOT signatures.
/// </summary>
public static unsafe partial class HipDriver
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

    /// <summary>ROCm HIP device attribute: number of multiprocessors / compute units (<c>hipDeviceAttributeMultiprocessorCount</c>).</summary>
    public const int HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63;

    public const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    static HipDriver()
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
                loaded = (NativeLibrary.TryLoad("amdhip64.dll", out IntPtr handle) ||
                          NativeLibrary.TryLoad("amdhip64_6.dll", out handle) ||
                          NativeLibrary.TryLoad("amdhip64_7.dll", out handle)) && handle != IntPtr.Zero;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                loaded = (NativeLibrary.TryLoad("libamdhip64.so", out IntPtr handle) ||
                          NativeLibrary.TryLoad("libamdhip64.so.7", out handle) ||
                          NativeLibrary.TryLoad("libamdhip64.so.6", out handle) ||
                          NativeLibrary.TryLoad("/opt/rocm/lib/libamdhip64.so", out handle) ||
                          NativeLibrary.TryLoad("/opt/rocm-7.2.0/lib/libamdhip64.so", out handle)) && handle != IntPtr.Zero;
            }

            if (!loaded)
                return false;

            return Init(0) == 0 && GetDeviceCount(out int devCount) == 0 && devCount > 0;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    [LibraryImport(HipLib, EntryPoint = "hipInit")]
    public static partial int Init(uint flags);

    [LibraryImport(HipLib, EntryPoint = "hipGetDeviceCount")]
    public static partial int GetDeviceCount(out int count);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceGetName")]
    public static partial int DeviceGetName(byte* name, int len, int dev);

    public static int DeviceGetName(byte[] name, int len, int dev)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentOutOfRangeException.ThrowIfNegative(len);
        int safeLen = Math.Min(len, name.Length);
        fixed (byte* pName = name)
        {
            return DeviceGetName(pName, safeLen, dev);
        }
    }

    [LibraryImport(HipLib, EntryPoint = "hipSetDevice")]
    public static partial int SetDevice(int device);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceSynchronize")]
    public static partial int DeviceSynchronize();

    [LibraryImport(HipLib, EntryPoint = "hipMalloc")]
    public static partial int Malloc(out IntPtr dptr, nuint size);

    [LibraryImport(HipLib, EntryPoint = "hipFree")]
    public static partial int Free(IntPtr dptr);

    [LibraryImport(HipLib, EntryPoint = "hipHostMalloc")]
    public static partial int HostMalloc(out IntPtr pptr, nuint size, uint flags);

    [LibraryImport(HipLib, EntryPoint = "hipHostFree")]
    public static partial int HostFree(IntPtr ptr);

    [LibraryImport(HipLib, EntryPoint = "hipHostGetDevicePointer")]
    public static partial int HostGetDevicePointer(out IntPtr devPtr, IntPtr hostPtr, uint flags);

    [LibraryImport(HipLib, EntryPoint = "hipMemcpy")]
    public static partial int Memcpy(IntPtr dst, IntPtr src, nuint size, int kind);

    [LibraryImport(HipLib, EntryPoint = "hipStreamCreate")]
    public static partial int StreamCreate(out IntPtr pStream);

    [LibraryImport(HipLib, EntryPoint = "hipStreamSynchronize")]
    public static partial int StreamSynchronize(IntPtr stream);

    [LibraryImport(HipLib, EntryPoint = "hipStreamDestroy")]
    public static partial int StreamDestroy(IntPtr stream);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceGetAttribute")]
    public static partial int DeviceGetAttribute(out int pi, int attr, int device);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceTotalMem")]
    public static partial int DeviceTotalMem(out nuint bytes, int device);

    public static string GetDeviceName(int device)
    {
        byte* buf = stackalloc byte[256];
        NativeMemory.Clear(buf, 256);
        int res = DeviceGetName(buf, 256, device);
        Check(res, nameof(DeviceGetName));
        int len = 0;
        while (len < 256 && buf[len] != 0) len++;
        return Encoding.ASCII.GetString(buf, len);
    }

    public static int GetMultiprocessorCount(int device)
    {
        int res = DeviceGetAttribute(out int count, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        if (res == 0 && count > 0)
            return count;

        // Only fall back when the driver returns a non-zero status for hipDeviceAttributeMultiprocessorCount (63).
        byte* buf = stackalloc byte[256];
        NativeMemory.Clear(buf, 256);
        if (DeviceGetName(buf, 256, device) != 0)
            return 0;

        int len = 0;
        while (len < 256 && buf[len] != 0) len++;
        string name = Encoding.ASCII.GetString(buf, len).ToLowerInvariant();
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
        return 0;
    }

    public static nuint GetTotalMemory(int device)
    {
        return DeviceTotalMem(out nuint mem, device) == 0 && mem > 0 ? mem : 0;
    }

    public static void Check(int res, string op)
    {
        if (res != 0)
        {
            throw new InvalidOperationException($"AMD HIP Driver Error during '{op}': code {res}");
        }
    }
}
