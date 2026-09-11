using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Glacier.Gpu.Compilation;

/// <summary>
/// Compiles PTX assembly into hardware SASS binaries (.cubin) using ptxas or driver JIT.
/// </summary>
public static class KernelCompiler
{
    private static readonly string[] PossiblePtxasPaths =
    {
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\ptxas.exe",
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe",
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe",
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\ptxas.exe",
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\ptxas.exe",
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\ptxas.exe",
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\ptxas.exe",
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\ptxas.exe",
        "ptxas.exe"
    };

    public static string? LocatePtxas()
    {
        foreach (var path in PossiblePtxasPaths)
        {
            if (File.Exists(path)) return path;
        }
        return null;
    }

    /// <summary>
    /// Compiles PTX into SASS cubin machine code. If ptxas is unavailable, returns UTF8 PTX bytes for driver JIT.
    /// </summary>
    public static byte[] CompileOrPreparePtx(string ptxCode, string targetArch = "sm_89")
    {
        if (!string.IsNullOrWhiteSpace(targetArch) && targetArch.StartsWith("sm_"))
        {
            ptxCode = System.Text.RegularExpressions.Regex.Replace(ptxCode, @"\.target\s+sm_\d+", $".target {targetArch}");
        }

        string? ptxas = LocatePtxas();
        if (ptxas == null)
        {
            // Fallback: driver-level JIT directly via cuModuleLoadData
            return Encoding.UTF8.GetBytes(ptxCode + "\0");
        }

        string tempDir = Path.Combine(Path.GetTempPath(), "GlacierGpu_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        string ptxFile = Path.Combine(tempDir, "kernel.ptx");
        string cubinFile = Path.Combine(tempDir, "kernel.cubin");

        try
        {
            File.WriteAllText(ptxFile, ptxCode);

            var psi = new ProcessStartInfo
            {
                FileName = ptxas,
                Arguments = $"--gpu-name={targetArch} -o \"{cubinFile}\" \"{ptxFile}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var proc = Process.Start(psi);
            if (proc != null)
            {
                string stdErr = proc.StandardError.ReadToEnd();
                string stdOut = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();

                if (proc.ExitCode == 0 && File.Exists(cubinFile))
                {
                    return File.ReadAllBytes(cubinFile);
                }
            }
        }
        catch
        {
            // Silently fall back to driver JIT
        }
        finally
        {
            try
            {
                if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
            }
            catch { }
        }

        return Encoding.UTF8.GetBytes(ptxCode + "\0");
    }
}
