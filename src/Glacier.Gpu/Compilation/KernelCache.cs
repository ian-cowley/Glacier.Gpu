// <copyright file="KernelCache.cs" company="Glacier High-Performance Ecosystem">
// Copyright (c) 2026 Ian Cowley. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace Glacier.Gpu.Compilation;

/// <summary>
/// On-device machine code artifact cache stored in %LOCALAPPDATA%\Glacier\GpuCache.
/// Uses full 256-bit SHA-256 keys, separate <c>.cubin</c> (ELF) vs <c>.ptx</c> storage,
/// path-traversal validation, and atomic temp-file writes.
/// </summary>
public static class KernelCache
{
    private static readonly string CacheBaseDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Glacier",
        "GpuCache");

    public static byte[] GetOrCompile(string targetArch, string kernelName, string ptxSource)
        => GetOrCompile(targetArch, kernelName, ptxSource, CacheBaseDir);

    internal static byte[] GetOrCompile(string targetArch, string kernelName, string ptxSource, string cacheBaseDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheBaseDir);
        ArgumentNullException.ThrowIfNull(ptxSource);

        string safeArch = ValidateAndSanitizeSegment(targetArch, nameof(targetArch));
        string safeKernel = ValidateAndSanitizeSegment(kernelName, nameof(kernelName));

        string baseFull = Path.GetFullPath(cacheBaseDir);
        string archDir = Path.GetFullPath(Path.Combine(baseFull, safeArch));
        string expectedPrefix = baseFull.EndsWith(Path.DirectorySeparatorChar)
            ? baseFull
            : baseFull + Path.DirectorySeparatorChar;
        if (!archDir.StartsWith(expectedPrefix, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Target architecture '{targetArch}' resolves outside the kernel cache directory.", nameof(targetArch));
        }

        Directory.CreateDirectory(archDir);

        string normalizedPtx = KernelCompiler.NormalizePtxDirectives(ptxSource, safeArch);
        string hash = ComputeCacheKey(safeArch, normalizedPtx);

        string cubinFile = Path.Combine(archDir, $"{safeKernel}_{hash}.cubin");
        string ptxFile = Path.Combine(archDir, $"{safeKernel}_{hash}.ptx");

        if (File.Exists(cubinFile))
        {
            try
            {
                byte[] cached = File.ReadAllBytes(cubinFile);
                if (HasElfMagic(cached))
                {
                    return cached;
                }
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        bool hasPtxas = KernelCompiler.LocatePtxas() is not null;
        if (!hasPtxas && File.Exists(ptxFile))
        {
            try
            {
                byte[] cached = File.ReadAllBytes(ptxFile);
                if (cached.Length > 0 && !HasElfMagic(cached))
                {
                    return cached;
                }
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        byte[] compiled = KernelCompiler.CompileOrPreparePtx(normalizedPtx, safeArch);
        string targetFile = HasElfMagic(compiled) ? cubinFile : ptxFile;
        WriteAtomic(archDir, safeKernel, hash, targetFile, compiled);

        return compiled;
    }

    internal static string ComputeCacheKey(string targetArch, string normalizedPtxSource)
    {
        string hashInput = $"{targetArch}\n{normalizedPtxSource}";
        byte[] hashBytes = SHA256.HashData(Encoding.UTF8.GetBytes(hashInput));
        return Convert.ToHexString(hashBytes);
    }

    internal static bool HasElfMagic(ReadOnlySpan<byte> data)
        => data.Length >= 4 &&
           data[0] == 0x7F &&
           data[1] == (byte)'E' &&
           data[2] == (byte)'L' &&
           data[3] == (byte)'F';

    internal static string ValidateAndSanitizeSegment(string value, string paramName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value, paramName);

        if (value == "." ||
            value.Contains("..", StringComparison.Ordinal) ||
            value.IndexOfAny(['/', '\\', ':', '\0']) >= 0)
        {
            throw new ArgumentException($"Invalid path characters in '{value}'.", paramName);
        }

        char[] invalidChars = Path.GetInvalidFileNameChars();
        if (value.IndexOfAny(invalidChars) >= 0)
        {
            throw new ArgumentException($"Invalid filename characters in '{value}'.", paramName);
        }

        Span<char> sanitized = value.Length <= 128 ? stackalloc char[value.Length] : new char[value.Length];
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            bool ok = (c >= 'a' && c <= 'z') ||
                      (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') ||
                      c == '.' || c == '_' || c == '-';
            sanitized[i] = ok ? c : '_';
        }

        string result = new string(sanitized);
        if (string.IsNullOrWhiteSpace(result) || result == "." || result.Contains("..", StringComparison.Ordinal))
        {
            throw new ArgumentException($"Invalid cache identifier '{value}'.", paramName);
        }

        return result;
    }

    private static void WriteAtomic(string archDir, string safeKernelName, string hash, string targetFile, byte[] data)
    {
        string tempFile = Path.Combine(archDir, $"{safeKernelName}_{hash}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (var fs = new FileStream(tempFile, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536))
            {
                fs.Write(data);
                fs.Flush(flushToDisk: true);
            }

            File.Move(tempFile, targetFile, overwrite: true);
        }
        catch (IOException)
        {
            // Non-fatal cache persistence failure (e.g., concurrent write or read-only volume).
        }
        catch (UnauthorizedAccessException)
        {
            // Non-fatal cache persistence failure on restricted directories.
        }
        finally
        {
            try
            {
                if (File.Exists(tempFile))
                {
                    File.Delete(tempFile);
                }
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }
}
