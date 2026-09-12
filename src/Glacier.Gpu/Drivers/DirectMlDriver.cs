using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Glacier.Gpu.Drivers;

/// <summary>
/// Direct P/Invoke bindings for Microsoft DirectML (DirectML.dll) and DirectX 12 (d3d12.dll).
/// Native Windows GPU compute driver for AMD Radeon, Intel Arc/Xe, and NVIDIA GPUs.
/// Zero external C++ runtime dependencies; ships as part of Windows 10/11.
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
        var list = new List<DxgiAdapterInfo>();
        if (!OperatingSystem.IsWindows()) return list;

        var factoryGuid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387"); // IID_IDXGIFactory1
        int hr = CreateDXGIFactory1(ref factoryGuid, out IntPtr factoryPtr);
        if (hr != 0 || factoryPtr == IntPtr.Zero) return list;

        try
        {
            var factory = (IDXGIFactory1)Marshal.GetObjectForIUnknown(factoryPtr);
            uint index = 0;

            while (true)
            {
                hr = factory.EnumAdapters1(index, out IDXGIAdapter1 adapter);
                if (hr != 0 || adapter == null) break;

                try
                {
                    hr = adapter.GetDesc1(out DXGI_ADAPTER_DESC1 desc);
                    if (hr == 0)
                    {
                        // Skip software rasterizer (flags & 2)
                        if ((desc.Flags & 2) == 0)
                        {
                            string descStr = desc.Description.Trim();
                            string lower = descStr.ToLowerInvariant();
                            bool isIntegrated = lower.Contains("graphics") || lower.Contains("890m") ||
                                                lower.Contains("780m") || lower.Contains("680m") ||
                                                lower.Contains("iris") || lower.Contains("uhd") ||
                                                (desc.DedicatedVideoMemory == 0 && desc.SharedSystemMemory > 0);

                            list.Add(new DxgiAdapterInfo
                            {
                                Index = (int)index,
                                Description = descStr,
                                VendorId = desc.VendorId,
                                DedicatedVramBytes = (ulong)desc.DedicatedVideoMemory,
                                SharedSystemMemoryBytes = (ulong)desc.SharedSystemMemory,
                                IsIntegrated = isIntegrated
                            });
                        }
                    }
                }
                finally
                {
                    Marshal.ReleaseComObject(adapter);
                }
                index++;
            }
        }
        catch { }
        finally
        {
            Marshal.Release(factoryPtr);
        }

        return list;
    }

    [DllImport(DxgiLib, ExactSpelling = true)]
    private static extern int CreateDXGIFactory1(ref Guid riid, out IntPtr ppFactory);

    [ComImport]
    [Guid("770aae78-f26f-4dba-a829-253c83d1b387")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIFactory1
    {
        int SetPrivateData(ref Guid Name, uint DataSize, IntPtr pData);
        int SetPrivateDataInterface(ref Guid Name, [MarshalAs(UnmanagedType.IUnknown)] object pUnknown);
        int GetPrivateData(ref Guid Name, ref uint pDataSize, IntPtr pData);
        int GetParent(ref Guid riid, out IntPtr ppParent);
        int EnumAdapters(uint Adapter, out IntPtr ppAdapter);
        int MakeWindowAssociation(IntPtr WindowHandle, uint Flags);
        int GetWindowAssociation(out IntPtr pWindowHandle);
        int CreateSwapChain(IntPtr pDevice, IntPtr pDesc, out IntPtr ppSwapChain);
        int CreateSoftwareAdapter(IntPtr Module, out IntPtr ppAdapter);

        [PreserveSig]
        int EnumAdapters1(uint Adapter, out IDXGIAdapter1 ppAdapter);
        [PreserveSig]
        int IsCurrent();
    }

    [ComImport]
    [Guid("29038f61-3839-4626-91fd-086879011a05")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIAdapter1
    {
        int SetPrivateData(ref Guid Name, uint DataSize, IntPtr pData);
        int SetPrivateDataInterface(ref Guid Name, [MarshalAs(UnmanagedType.IUnknown)] object pUnknown);
        int GetPrivateData(ref Guid Name, ref uint pDataSize, IntPtr pData);
        int GetParent(ref Guid riid, out IntPtr ppParent);

        [PreserveSig]
        int EnumOutputs(uint Output, out IntPtr ppOutput);
        [PreserveSig]
        int GetDesc(IntPtr pDesc);
        [PreserveSig]
        int CheckInterfaceSupport(ref Guid InterfaceName, out long pUMDVersion);

        [PreserveSig]
        int GetDesc1(out DXGI_ADAPTER_DESC1 pDesc);
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct DXGI_ADAPTER_DESC1
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public nuint DedicatedVideoMemory;
        public nuint DedicatedSystemMemory;
        public nuint SharedSystemMemory;
        public long AdapterLuid;
        public uint Flags;
    }
}
