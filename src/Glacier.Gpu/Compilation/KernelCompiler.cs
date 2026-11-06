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
        "/usr/local/cuda/bin/ptxas",
        "/usr/bin/ptxas",
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
    /// Aligns both <c>.version</c> and <c>.target</c> PTX header directives with <paramref name="targetArch"/>
    /// so PTX ISA minimum version requirements are satisfied across Turing (sm_75), Ampere (sm_80/86), Ada (sm_89), Hopper (sm_90), and Blackwell (sm_100/120).
    /// </summary>
    public static string NormalizePtxDirectives(string ptxCode, string targetArch)
    {
        ArgumentNullException.ThrowIfNull(ptxCode);
        if (string.IsNullOrWhiteSpace(targetArch) || !targetArch.StartsWith("sm_", StringComparison.Ordinal))
            return ptxCode;

        string requiredPtxVersion = "7.5";
        if (int.TryParse(targetArch.AsSpan(3), out int smNum))
        {
            if (smNum >= 100)
                requiredPtxVersion = "8.7";
            else if (smNum >= 89)
                requiredPtxVersion = "8.0";
            else if (smNum >= 75)
                requiredPtxVersion = "7.5";
            else
                requiredPtxVersion = "7.0";
        }

        ptxCode = ReplaceHeaderDirective(ptxCode, ".version ", requiredPtxVersion);
        ptxCode = ReplaceHeaderDirective(ptxCode, ".target ", targetArch);
        return ptxCode;
    }

    /// <summary>
    /// Compiles PTX into SASS cubin machine code. If ptxas is unavailable, returns null-terminated UTF8 PTX bytes for driver JIT.
    /// </summary>
    public static byte[] CompileOrPreparePtx(string ptxCode, string targetArch = "sm_89")
    {
        ArgumentNullException.ThrowIfNull(ptxCode);
        ptxCode = NormalizePtxDirectives(ptxCode, targetArch);

        string? ptxas = LocatePtxas();
        if (ptxas == null)
        {
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
                var stdOutTask = proc.StandardOutput.ReadToEndAsync();
                var stdErrTask = proc.StandardError.ReadToEndAsync();

                if (proc.WaitForExit(10_000))
                {
                    _ = stdOutTask.GetAwaiter().GetResult();
                    _ = stdErrTask.GetAwaiter().GetResult();
                    if (proc.ExitCode == 0 && File.Exists(cubinFile))
                    {
                        return File.ReadAllBytes(cubinFile);
                    }
                }
                else
                {
                    try { proc.Kill(entireProcessTree: true); } catch { }
                }
            }
        }
        catch
        {
            // Fall back cleanly to CUDA driver JIT via cuModuleLoadData
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

    private static string ReplaceHeaderDirective(string ptxCode, string directivePrefix, string newValue)
    {
        int idx = ptxCode.IndexOf(directivePrefix, StringComparison.Ordinal);
        if (idx < 0) return ptxCode;

        int lineEnd = ptxCode.IndexOfAny(['\r', '\n'], idx);
        if (lineEnd < 0)
        {
            return string.Concat(ptxCode.AsSpan(0, idx), directivePrefix, newValue);
        }

        return string.Concat(ptxCode.AsSpan(0, idx), directivePrefix, newValue, ptxCode.AsSpan(lineEnd));
    }
}
