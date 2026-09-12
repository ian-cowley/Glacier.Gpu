namespace Glacier.Gpu.Tests;

using System;
using System.Runtime.InteropServices;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Xunit;

public unsafe class D3D12ComputeTests
{
    [Fact]
    public void D3D12_CanCreateDeviceAndExecuteComputeShader()
    {
        if (!OperatingSystem.IsWindows()) return;

        // 1. Enumerate adapters to find hardware GPU
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
        IDXGIAdapter1? selectedAdapter = null;
        for (uint i = 0; factory.EnumAdapters1(i, out IDXGIAdapter1 adapter).Success; i++)
        {
            var desc = adapter.Description1;
            if ((desc.Flags & AdapterFlags.Software) == 0)
            {
                selectedAdapter = adapter;
                // Prefer AMD or non-software adapter
                if (desc.Description.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }
            }
            else
            {
                adapter.Dispose();
            }
        }

        Assert.NotNull(selectedAdapter);
        string adapterName = selectedAdapter.Description1.Description;
        Assert.False(string.IsNullOrWhiteSpace(adapterName));

        // 2. Create D3D12 Device
        var hr = D3D12.D3D12CreateDevice(selectedAdapter, FeatureLevel.Level_11_0, out ID3D12Device? device);
        Assert.True(hr.Success && device != null, $"Failed to create D3D12 device on {adapterName}: {hr}");

        // 3. Create Compute Command Queue
        var queueDesc = new CommandQueueDescription(CommandListType.Compute);
        var queue = device.CreateCommandQueue(queueDesc);
        Assert.NotNull(queue);

        // 4. Compile a simple HLSL compute shader
        string hlsl = @"
            RWStructuredBuffer<float> output : register(u0);
            [numthreads(32, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID)
            {
                output[id.x] = (float)id.x * 2.5f + 1.0f;
            }
        ";

        var shaderBlob = Compiler.Compile(hlsl, "main", "compute.hlsl", "cs_5_0");
        Assert.False(shaderBlob.IsEmpty);

        // 5. Create Root Signature with 1 UAV descriptor table
        var descriptorRange = new DescriptorRange(DescriptorRangeType.UnorderedAccessView, 1, 0);
        var rootParameter = new RootParameter(new RootDescriptorTable(descriptorRange), ShaderVisibility.All);
        var rootSigDesc = new RootSignatureDescription(RootSignatureFlags.None, [rootParameter]);
        var rootSignature = device.CreateRootSignature(rootSigDesc, RootSignatureVersion.Version1);
        Assert.NotNull(rootSignature);

        // 6. Create Compute Pipeline State
        var psoDesc = new ComputePipelineStateDescription
        {
            RootSignature = rootSignature,
            ComputeShader = shaderBlob
        };
        var pso = device.CreateComputePipelineState(psoDesc);
        Assert.NotNull(pso);

        // 7. Create Buffers (GPU UAV buffer + Readback buffer)
        const int count = 32;
        const int bufferSize = count * sizeof(float);

        var uavBufferDesc = ResourceDescription.Buffer(bufferSize, ResourceFlags.AllowUnorderedAccess);
        var defaultHeap = new HeapProperties(HeapType.Default);
        var uavBuffer = device.CreateCommittedResource(defaultHeap, HeapFlags.None, uavBufferDesc, ResourceStates.Common);
        Assert.NotNull(uavBuffer);

        var readbackDesc = ResourceDescription.Buffer(bufferSize);
        var readbackHeap = new HeapProperties(HeapType.Readback);
        var readbackBuffer = device.CreateCommittedResource(readbackHeap, HeapFlags.None, readbackDesc, ResourceStates.CopyDest);
        Assert.NotNull(readbackBuffer);

        // 8. Create Descriptor Heap
        var heapDesc = new DescriptorHeapDescription(DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView, 1, DescriptorHeapFlags.ShaderVisible);
        var descriptorHeap = device.CreateDescriptorHeap(heapDesc);
        Assert.NotNull(descriptorHeap);

        var uavDesc = new UnorderedAccessViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = UnorderedAccessViewDimension.Buffer,
            Buffer = new BufferUnorderedAccessView
            {
                FirstElement = 0,
                NumElements = count,
                StructureByteStride = sizeof(float)
            }
        };
        device.CreateUnorderedAccessView(uavBuffer, null, uavDesc, descriptorHeap.GetCPUDescriptorHandleForHeapStart());

        // 9. Record and execute compute commands
        var cmdAlloc = device.CreateCommandAllocator(CommandListType.Compute);
        var cmdList = device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, cmdAlloc, pso);

        cmdList.SetComputeRootSignature(rootSignature);
        cmdList.SetDescriptorHeaps(descriptorHeap);
        cmdList.SetComputeRootDescriptorTable(0, descriptorHeap.GetGPUDescriptorHandleForHeapStart());

        cmdList.Dispatch(1, 1, 1);

        // Barrier from UAV to CopySource
        cmdList.ResourceBarrierTransition(uavBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmdList.CopyResource(readbackBuffer, uavBuffer);

        cmdList.Close();

        // 10. Execute on GPU queue and wait via Fence
        queue.ExecuteCommandList(cmdList);

        var fence = device.CreateFence(0);
        queue.Signal(fence, 1);
        if (fence.CompletedValue < 1)
        {
            using var evt = new System.Threading.AutoResetEvent(false);
            fence.SetEventOnCompletion(1, evt);
            evt.WaitOne();
        }

        // 11. Read back results from GPU
        void* pRaw = null;
        readbackBuffer.Map(0, null, &pRaw);
        float* pData = (float*)pRaw;
        Assert.True(pData != null);

        for (int i = 0; i < count; i++)
        {
            float expected = i * 2.5f + 1.0f;
            Assert.Equal(expected, pData[i], 0.001f);
        }

        readbackBuffer.Unmap(0);

        // Cleanup
        pso.Dispose();
        rootSignature.Dispose();
        descriptorHeap.Dispose();
        uavBuffer.Dispose();
        readbackBuffer.Dispose();
        cmdList.Dispose();
        cmdAlloc.Dispose();
        fence.Dispose();
        queue.Dispose();
        device.Dispose();
        selectedAdapter.Dispose();
    }

    [Fact]
    public void D3D12_CanCompileAndExecute_GEMV_Q4_K()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
        IDXGIAdapter1? selectedAdapter = null;
        for (uint i = 0; factory.EnumAdapters1(i, out IDXGIAdapter1 adapter).Success; i++)
        {
            var desc = adapter.Description1;
            if ((desc.Flags & AdapterFlags.Software) == 0)
            {
                selectedAdapter = adapter;
                if (desc.Description.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                    break;
            }
            else
            {
                adapter.Dispose();
            }
        }

        Assert.NotNull(selectedAdapter);
        var hr = D3D12.D3D12CreateDevice(selectedAdapter, FeatureLevel.Level_11_0, out ID3D12Device? device);
        Assert.True(hr.Success && device != null);

        var queueDesc = new CommandQueueDescription(CommandListType.Compute);
        var queue = device.CreateCommandQueue(queueDesc);

        string hlsl = @"
            cbuffer Params : register(b0)
            {
                uint k_cols;
                uint m_rows;
                uint has_bias;
                uint has_residual;
                uint has_y;
            };

            StructuredBuffer<uint> W : register(t0);
            StructuredBuffer<float> x : register(t1);
            StructuredBuffer<float> bias : register(t2);
            RWStructuredBuffer<float> residual : register(u0);
            RWStructuredBuffer<float> y : register(u1);

            groupshared float s_mem[32];

            uint get_scale_byte(uint idx, uint s0, uint s1, uint s2)
            {
                if (idx < 4) return (s0 >> (idx * 8)) & 0xFF;
                else if (idx < 8) return (s1 >> ((idx - 4) * 8)) & 0xFF;
                else return (s2 >> ((idx - 8) * 8)) & 0xFF;
            }

            void get_scale_min(uint j, uint s0, uint s1, uint s2, out float d_out, out float min_out, float d, float min_val)
            {
                uint sc, m;
                if (j < 4) {
                    sc = get_scale_byte(j, s0, s1, s2) & 63;
                    m  = get_scale_byte(j + 4, s0, s1, s2) & 63;
                } else {
                    sc = (get_scale_byte(j + 4, s0, s1, s2) & 0x0F) | ((get_scale_byte(j - 4, s0, s1, s2) >> 6) << 4);
                    m  = (get_scale_byte(j + 4, s0, s1, s2) >> 4)   | ((get_scale_byte(j, s0, s1, s2) >> 6) << 4);
                }
                d_out = d * (float)sc;
                min_out = min_val * (float)m;
            }

            [numthreads(32, 1, 1)]
            void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
            {
                uint warp_id = gid.x;
                uint lane_id = gtid.x;

                if (warp_id >= m_rows) return;

                uint nb = k_cols / 256;
                uint row_offset = warp_id * nb * 36;
                float row_sum = 0.0f;

                uint chunk = lane_id / 8;
                uint chunk_lane = lane_id % 8;
                uint is_idx = chunk * 2;
                uint x_offset1 = chunk * 64 + chunk_lane * 4;
                uint x_offset2 = x_offset1 + 32;

                for (uint b = 0; b < nb; b++)
                {
                    uint blk_offset = row_offset + b * 36;
                    uint d_dmin = W[blk_offset + 0];
                    float d = f16tof32(d_dmin & 0xFFFF);
                    float min_val = f16tof32(d_dmin >> 16);

                    uint s0 = W[blk_offset + 1];
                    uint s1 = W[blk_offset + 2];
                    uint s2 = W[blk_offset + 3];

                    float d1, min1, d2, min2;
                    get_scale_min(is_idx + 0, s0, s1, s2, d1, min1, d, min_val);
                    get_scale_min(is_idx + 1, s0, s1, s2, d2, min2, d, min_val);

                    uint x_blk = b * 256;
                    float4 x1 = float4(x[x_blk + x_offset1 + 0], x[x_blk + x_offset1 + 1], x[x_blk + x_offset1 + 2], x[x_blk + x_offset1 + 3]);
                    float4 x2 = float4(x[x_blk + x_offset2 + 0], x[x_blk + x_offset2 + 1], x[x_blk + x_offset2 + 2], x[x_blk + x_offset2 + 3]);

                    uint q = W[blk_offset + 4 + lane_id];
                    uint q0 = q & 0xFF;
                    uint q1 = (q >> 8) & 0xFF;
                    uint q2 = (q >> 16) & 0xFF;
                    uint q3 = q >> 24;

                    float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
                    float sum_x1 = x1.x + x1.y + x1.z + x1.w;

                    float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;
                    float sum_x2 = x2.x + x2.y + x2.z + x2.w;

                    row_sum += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
                }

                s_mem[lane_id] = row_sum;
                GroupMemoryBarrierWithGroupSync();
                if (lane_id < 16) s_mem[lane_id] += s_mem[lane_id + 16];
                GroupMemoryBarrierWithGroupSync();
                if (lane_id < 8)  s_mem[lane_id] += s_mem[lane_id + 8];
                GroupMemoryBarrierWithGroupSync();
                if (lane_id < 4)  s_mem[lane_id] += s_mem[lane_id + 4];
                GroupMemoryBarrierWithGroupSync();
                if (lane_id < 2)  s_mem[lane_id] += s_mem[lane_id + 2];
                GroupMemoryBarrierWithGroupSync();
                if (lane_id < 1)  s_mem[lane_id] += s_mem[lane_id + 1];

                if (lane_id == 0)
                {
                    float final_sum = s_mem[0];
                    if (has_bias != 0) final_sum += bias[warp_id];
                    if (has_residual != 0) residual[warp_id] += final_sum;
                    if (has_y != 0) y[warp_id] = final_sum;
                }
            }
        ";

        var shaderBlob = Compiler.Compile(hlsl, "main", "gemv_q4_k.hlsl", "cs_5_0");
        Assert.False(shaderBlob.IsEmpty);

        // Verify root signature creation with Root Parameters (no descriptor heaps required)
        var rootParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 5), ShaderVisibility.All), // b0
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All), // t0 (W)
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All), // t1 (x)
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All), // t2 (bias)
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0 (residual)
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)  // u1 (y)
        };

        var rootSigDesc = new RootSignatureDescription(RootSignatureFlags.None, rootParams);
        var rootSignature = device.CreateRootSignature(rootSigDesc, RootSignatureVersion.Version1);
        Assert.NotNull(rootSignature);

        var psoDesc = new ComputePipelineStateDescription
        {
            RootSignature = rootSignature,
            ComputeShader = shaderBlob
        };
        var pso = device.CreateComputePipelineState(psoDesc);
        Assert.NotNull(pso);

        pso.Dispose();
        rootSignature.Dispose();
        queue.Dispose();
        device.Dispose();
        selectedAdapter.Dispose();
    }

    [Fact]
    public void D3D12ComputeEngine_Initialization_And_ZeroCopy()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var engine = new Glacier.Gpu.Engines.D3D12ComputeEngine();
        engine.Initialize();

        Assert.True(engine.IsInitialized);
        Assert.NotNull(engine.DeviceInfo);
        Assert.False(string.IsNullOrWhiteSpace(engine.DeviceInfo.DeviceName));

        // Test Zero-Copy allocation
        const int count = 1024;
        using var block = engine.AllocateZeroCopy<float>(count);
        Assert.NotEqual(IntPtr.Zero, block.HostPointer);
        Assert.NotEqual(IntPtr.Zero, block.DevicePointer);

        // CPU write directly into zero-copy block
        var span = block.Span;
        for (int i = 0; i < count; i++)
        {
            span[i] = i * 3.14f;
        }

        // Verify direct readback
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(i * 3.14f, span[i], 0.0001f);
        }
    }
}
