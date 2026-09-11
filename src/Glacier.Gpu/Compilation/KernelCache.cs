using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace Glacier.Gpu.Compilation;

/// <summary>
/// On-device machine code artifact cache stored in %LOCALAPPDATA%\Glacier\GpuCache.
/// Bypasses compilation on subsequent executions (&lt; 2 ms load time).
/// </summary>
public static class KernelCache
{
    private static readonly string CacheBaseDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Glacier",
        "GpuCache"
    );

    public static byte[] GetOrCompile(string targetArch, string kernelName, string ptxSource)
    {
        string archDir = Path.Combine(CacheBaseDir, targetArch);
        Directory.CreateDirectory(archDir);

        using var sha = SHA256.Create();
        string hash = Convert.ToHexString(sha.ComputeHash(Encoding.UTF8.GetBytes(ptxSource)))[..16];
        string cacheFile = Path.Combine(archDir, $"{kernelName}_{hash}.cubin");

        if (File.Exists(cacheFile))
        {
            try
            {
                return File.ReadAllBytes(cacheFile);
            }
            catch { }
        }

        byte[] compiled = KernelCompiler.CompileOrPreparePtx(ptxSource, targetArch);
        try
        {
            File.WriteAllBytes(cacheFile, compiled);
        }
        catch { }

        return compiled;
    }
}
