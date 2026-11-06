using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Glacier.Gpu.Drivers;

/// <summary>
/// Direct hardware adapter discovery for Microsoft DirectML (DirectML.dll) and DirectX 12 (d3d12.dll).
/// Native Windows GPU compute probe for AMD Radeon, Intel Arc/Xe, and NVIDIA GPUs.
/// Uses Native AOT-safe DXGI enumeration without reflection or RCWs.
/// </summary>
public static class DirectMlDriver
{
    private const string DirectMlLib = "DirectML.dll";
    private const string D3D12Lib = "d3d12.dll";
    private const string DxgiLib = "dxgi.dll";

    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        if (!OperatingSystem.IsWindows()) return false;
        try
        {
            bool dmlOk = NativeLibrary.TryLoad(DirectMlLib, out IntPtr hDml) && hDml != IntPtr.Zero;
            bool d3dOk = NativeLibrary.TryLoad(D3D12Lib, out IntPtr hD3d) && hD3d != IntPtr.Zero;
            bool dxgiOk = NativeLibrary.TryLoad(DxgiLib, out IntPtr hDxgi) && hDxgi != IntPtr.Zero;
            return dmlOk && d3dOk && dxgiOk;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    public sealed record DxgiAdapterInfo
    {
        public int Index { get; init; }
        public string Description { get; init; } = "";
        public uint VendorId { get; init; }
        public ulong DedicatedVramBytes { get; init; }
        public ulong SharedSystemMemoryBytes { get; init; }
        public bool IsIntegrated { get; init; }
        public bool IsIntel => VendorId == 0x8086;
        public bool IsAmd => VendorId == 0x1002;
        public bool IsNvidia => VendorId == 0x10DE;
    }

    private static readonly Lazy<List<DxgiAdapterInfo>> _adapters = new(EnumerateAdapters);

    public static IReadOnlyList<DxgiAdapterInfo> GetAdapters() => _adapters.Value;

    public static bool HasHardwareAdapter => GetAdapters().Count > 0;

    public static bool HasIntelGpu
    {
        get
        {
            foreach (var a in GetAdapters())
                if (a.IsIntel) return true;
            return false;
        }
    }

    public static bool HasAmdGpu
    {
        get
        {
            foreach (var a in GetAdapters())
                if (a.IsAmd) return true;
            return false;
        }
    }

    public static bool HasNvidiaGpu
    {
        get
        {
            foreach (var a in GetAdapters())
                if (a.IsNvidia) return true;
            return false;
        }
    }

    public static DxgiAdapterInfo? GetFirstAdapterForVendor(uint vendorId)
    {
        foreach (var a in GetAdapters())
            if (a.VendorId == vendorId) return a;
        return null;
    }

    private static List<DxgiAdapterInfo> EnumerateAdapters()
    {
        if (!OperatingSystem.IsWindows())
            return new List<DxgiAdapterInfo>();

        return EnumerateAdaptersWindows();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static List<DxgiAdapterInfo> EnumerateAdaptersWindows()
    {
        var list = new List<DxgiAdapterInfo>();
        try
        {
            using var factory = Vortice.DXGI.DXGI.CreateDXGIFactory1<Vortice.DXGI.IDXGIFactory1>();
            for (uint index = 0; factory.EnumAdapters1(index, out Vortice.DXGI.IDXGIAdapter1 adapter).Success; index++)
            {
                using (adapter)
                {
                    var desc = adapter.Description1;
                    if ((desc.Flags & Vortice.DXGI.AdapterFlags.Software) != 0)
                        continue;

                    string descStr = desc.Description?.Trim() ?? string.Empty;
                    string lower = descStr.ToLowerInvariant();
                    ulong dedicatedVram = (ulong)desc.DedicatedVideoMemory;
                    ulong sharedMem = (ulong)desc.SharedSystemMemory;
                    bool isIntegrated = lower.Contains("graphics") || lower.Contains("890m") ||
                                        lower.Contains("780m") || lower.Contains("680m") ||
                                        lower.Contains("iris") || lower.Contains("uhd") ||
                                        (dedicatedVram == 0 && sharedMem > 0);

                    list.Add(new DxgiAdapterInfo
                    {
                        Index = (int)index,
                        Description = descStr,
                        VendorId = (uint)desc.VendorId,
                        DedicatedVramBytes = dedicatedVram,
                        SharedSystemMemoryBytes = sharedMem,
                        IsIntegrated = isIntegrated
                    });
                }
            }
        }
        catch
        {
            // Headless or unsupported DXGI environment
        }

        return list;
    }
}
