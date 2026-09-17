namespace Glacier.Gpu.Compilation;

/// <summary>
/// Embedded hardware compute kernels in PTX and high-performance assembly.
/// </summary>
public static class KernelSources
{
    public const string VectorAddPtx = @"
.version 8.0
.target sm_89
.address_size 64

.visible .entry vector_add(
    .param .u64 d_a,
    .param .u64 d_b,
    .param .u64 d_c,
    .param .u32 n
)
{
    .reg .pred %p0;
    .reg .b32 %r0, %r1, %r2;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;

    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd2, %rd1, 2;

    ld.param.u64 %rd3, [d_a];
    add.s64 %rd4, %rd3, %rd2;
    ld.global.f32 %f1, [%rd4];

    ld.param.u64 %rd5, [d_b];
    add.s64 %rd6, %rd5, %rd2;
    ld.global.f32 %f2, [%rd6];

    add.f32 %f3, %f1, %f2;

    ld.param.u64 %rd7, [d_c];
    add.s64 %rd7, %rd7, %rd2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
";

    public const string VectorFmaPtx = @"
.version 8.0
.target sm_89
.address_size 64

.visible .entry vector_fma_benchmark(
    .param .u64 d_a,
    .param .u64 d_b,
    .param .u64 d_c,
    .param .u32 n,
    .param .u32 loop_count
)
{
    .reg .pred %p0, %p1;
    .reg .b32 %r0, %r1, %r2, %i, %loops;
    .reg .b64 %rd<8>;
    .reg .f32 %f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8, %f9, %f_scaleA, %f_scaleB;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;

    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd2, %rd1, 2;

    ld.param.u64 %rd3, [d_a];
    add.s64 %rd4, %rd3, %rd2;
    ld.global.f32 %f0, [%rd4];

    ld.param.u64 %rd5, [d_b];
    add.s64 %rd6, %rd5, %rd2;
    ld.global.f32 %f1, [%rd6];

    mov.f32 %f2, %f0;
    mov.f32 %f3, %f1;
    mov.f32 %f4, %f0;
    mov.f32 %f5, %f1;
    mov.f32 %f6, %f0;
    mov.f32 %f7, %f1;
    mov.f32 %f8, %f0;
    mov.f32 %f9, %f1;

    mov.f32 %f_scaleA, 1.00001;
    mov.f32 %f_scaleB, 0.99999;

    ld.param.u32 %loops, [loop_count];
    mov.u32 %i, 0;

LOOP:
    fma.rn.f32 %f2, %f2, %f_scaleA, %f3;
    fma.rn.f32 %f4, %f4, %f_scaleB, %f5;
    fma.rn.f32 %f6, %f6, %f_scaleA, %f7;
    fma.rn.f32 %f8, %f8, %f_scaleB, %f9;
    fma.rn.f32 %f3, %f3, %f_scaleA, %f2;
    fma.rn.f32 %f5, %f5, %f_scaleB, %f4;
    fma.rn.f32 %f7, %f7, %f_scaleA, %f6;
    fma.rn.f32 %f9, %f9, %f_scaleB, %f8;
    add.u32 %i, %i, 1;
    setp.lt.u32 %p1, %i, %loops;
    @%p1 bra LOOP;

    add.f32 %f2, %f2, %f3;
    add.f32 %f4, %f4, %f5;
    add.f32 %f6, %f6, %f7;
    add.f32 %f8, %f8, %f9;
    add.f32 %f2, %f2, %f4;
    add.f32 %f6, %f6, %f8;
    add.f32 %f0, %f2, %f6;

    ld.param.u64 %rd7, [d_c];
    add.s64 %rd7, %rd7, %rd2;
    st.global.f32 [%rd7], %f0;

DONE:
    ret;
}
";

    public const string PersistentWorkerPtx = @"
.version 8.0
.target sm_89
.address_size 64

.visible .entry persistent_ring_worker(
    .param .u64 task_ptr
)
{
    .shared .align 8 .u32 s_opcode;
    .shared .align 8 .u32 s_count;
    .shared .align 8 .u64 s_bufA;
    .shared .align 8 .u64 s_bufB;
    .shared .align 8 .u64 s_bufC;
    .shared .align 4 .f32 s_scalar;

    .reg .pred %p_is_worker, %p_has_task, %p_exit, %p_add, %p_fma, %p_done;
    .reg .b32 %r_tid, %opcode, %status, %count, %last_task_id, %curr_task_id, %r_idx, %r_stride;
    .reg .b64 %rd_task, %rd_bufA, %rd_bufB, %rd_bufC, %rd_offset, %rd_ptr;
    .reg .f32 %fA, %fB, %fC, %f_scalar;

    ld.param.u64 %rd_task, [task_ptr];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %last_task_id, 0;
    setp.ne.u32 %p_is_worker, %r_tid, 0;

POLL_LOOP:
    @%p_is_worker bra WAIT_FOR_POLL;

MASTER_POLL:
    ld.acquire.sys.global.u32 %curr_task_id, [%rd_task];
    setp.ne.u32 %p_has_task, %curr_task_id, %last_task_id;
    @!%p_has_task bra MASTER_POLL;

    mov.u32 %last_task_id, %curr_task_id;

BROADCAST_TASK:
    ld.acquire.sys.global.u32 %opcode, [%rd_task + 4];
    st.shared.u32 [s_opcode], %opcode;
    ld.global.u32 %count, [%rd_task + 8];
    st.shared.u32 [s_count], %count;
    ld.global.u64 %rd_bufA, [%rd_task + 16];
    st.shared.u64 [s_bufA], %rd_bufA;
    ld.global.u64 %rd_bufB, [%rd_task + 24];
    st.shared.u64 [s_bufB], %rd_bufB;
    ld.global.u64 %rd_bufC, [%rd_task + 32];
    st.shared.u64 [s_bufC], %rd_bufC;
    ld.global.f32 %f_scalar, [%rd_task + 48];
    st.shared.f32 [s_scalar], %f_scalar;

WAIT_FOR_POLL:
    bar.sync 0;

    ld.shared.u32 %opcode, [s_opcode];
    setp.eq.u32 %p_exit, %opcode, 99;
    @%p_exit bra EXIT_WORKER;

    ld.shared.u32 %count, [s_count];
    mov.u32 %r_idx, %r_tid;
    mov.u32 %r_stride, %ntid.x;

    setp.eq.u32 %p_add, %opcode, 1;
    @%p_add bra LOOP_ADD;

    setp.eq.u32 %p_fma, %opcode, 2;
    @%p_fma bra LOOP_FMA;

    bra WORK_COMPLETE;

LOOP_ADD:
    setp.ge.u32 %p_done, %r_idx, %count;
    @%p_done bra WORK_COMPLETE;

    cvt.u64.u32 %rd_offset, %r_idx;
    shl.b64 %rd_offset, %rd_offset, 2;

    ld.shared.u64 %rd_bufA, [s_bufA];
    ld.shared.u64 %rd_bufB, [s_bufB];
    ld.shared.u64 %rd_bufC, [s_bufC];

    add.s64 %rd_ptr, %rd_bufA, %rd_offset;
    ld.global.f32 %fA, [%rd_ptr];

    add.s64 %rd_ptr, %rd_bufB, %rd_offset;
    ld.global.f32 %fB, [%rd_ptr];

    add.f32 %fC, %fA, %fB;

    add.s64 %rd_ptr, %rd_bufC, %rd_offset;
    st.global.f32 [%rd_ptr], %fC;

    add.u32 %r_idx, %r_idx, %r_stride;
    bra LOOP_ADD;

LOOP_FMA:
    setp.ge.u32 %p_done, %r_idx, %count;
    @%p_done bra WORK_COMPLETE;

    cvt.u64.u32 %rd_offset, %r_idx;
    shl.b64 %rd_offset, %rd_offset, 2;

    ld.shared.u64 %rd_bufA, [s_bufA];
    ld.shared.u64 %rd_bufB, [s_bufB];
    ld.shared.u64 %rd_bufC, [s_bufC];
    ld.shared.f32 %f_scalar, [s_scalar];

    add.s64 %rd_ptr, %rd_bufA, %rd_offset;
    ld.global.f32 %fA, [%rd_ptr];

    add.s64 %rd_ptr, %rd_bufB, %rd_offset;
    ld.global.f32 %fB, [%rd_ptr];

    fma.rn.f32 %fC, %fA, %f_scalar, %fB;

    add.s64 %rd_ptr, %rd_bufC, %rd_offset;
    st.global.f32 [%rd_ptr], %fC;

    add.u32 %r_idx, %r_idx, %r_stride;
    bra LOOP_FMA;

WORK_COMPLETE:
    bar.sync 0;

    @%p_is_worker bra POLL_LOOP;

    membar.sys;
    mov.u32 %status, 1;
    st.global.volatile.u32 [%rd_task + 12], %status;
    bra POLL_LOOP;

EXIT_WORKER:
    ret;
}
";
}
