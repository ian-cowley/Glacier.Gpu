namespace Glacier.Gpu.Compilation;

/// <summary>
/// Embedded hardware compute kernels in PTX and high-performance assembly.
/// Targets baseline PTX 7.5 / sm_75 (Turing+) and scales dynamically via KernelCompiler.
/// </summary>
public static class KernelSources
{
    public const string VectorAddPtx = @"
.version 7.5
.target sm_75
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
.version 7.5
.target sm_75
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
.version 7.5
.target sm_75
.address_size 64

.visible .entry persistent_ring_worker(
    .param .u64 task_ptr,
    .param .u32 ring_mask,
    .param .u64 sync_ptr
)
{
    .shared .align 8 .u32 s_opcode;
    .shared .align 8 .u32 s_count;
    .shared .align 8 .u64 s_bufA;
    .shared .align 8 .u64 s_bufB;
    .shared .align 8 .u64 s_bufC;
    .shared .align 4 .f32 s_scalar;

    .reg .pred %p_is_worker, %p_has_task, %p_exit, %p_add, %p_fma, %p_done, %p_last, %p_wait;
    .reg .b32 %r_tid, %r_ntid, %r_cta_id, %r_nctas, %opcode, %status, %count, %expected_task_id, %curr_task_id, %r_idx, %r_stride, %r_slot, %r_mask, %r_prev, %r_arrived, %r_done_id;
    .reg .b64 %rd_task, %rd_sync, %rd_slot_ptr, %rd_slot_off, %rd_bufA, %rd_bufB, %rd_bufC, %rd_offset, %rd_ptr;
    .reg .f32 %fA, %fB, %fC, %f_scalar;

    ld.param.u64 %rd_task, [task_ptr];
    ld.param.u32 %r_mask, [ring_mask];
    ld.param.u64 %rd_sync, [sync_ptr];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_cta_id, %ctaid.x;
    mov.u32 %r_nctas, %nctaid.x;
    mov.u32 %r_slot, 0;
    mov.u32 %expected_task_id, 1;
    setp.ne.u32 %p_is_worker, %r_tid, 0;

POLL_LOOP:
    @%p_is_worker bra WAIT_FOR_POLL;

    cvt.u64.u32 %rd_slot_off, %r_slot;
    shl.b64 %rd_slot_off, %rd_slot_off, 6;
    add.u64 %rd_slot_ptr, %rd_task, %rd_slot_off;

MASTER_POLL:
    ld.acquire.sys.global.u32 %curr_task_id, [%rd_slot_ptr];
    setp.eq.u32 %p_has_task, %curr_task_id, %expected_task_id;
    @!%p_has_task bra MASTER_POLL;

BROADCAST_TASK:
    ld.global.u32 %opcode, [%rd_slot_ptr + 4];
    st.shared.u32 [s_opcode], %opcode;
    ld.global.u32 %count, [%rd_slot_ptr + 8];
    st.shared.u32 [s_count], %count;
    ld.global.u64 %rd_bufA, [%rd_slot_ptr + 16];
    st.shared.u64 [s_bufA], %rd_bufA;
    ld.global.u64 %rd_bufB, [%rd_slot_ptr + 24];
    st.shared.u64 [s_bufB], %rd_bufB;
    ld.global.u64 %rd_bufC, [%rd_slot_ptr + 32];
    st.shared.u64 [s_bufC], %rd_bufC;
    ld.global.f32 %f_scalar, [%rd_slot_ptr + 48];
    st.shared.f32 [s_scalar], %f_scalar;

WAIT_FOR_POLL:
    bar.sync 0;

    ld.shared.u32 %opcode, [s_opcode];
    setp.eq.u32 %p_exit, %opcode, 99;
    @%p_exit bra EXIT_WORKER;

    ld.shared.u32 %count, [s_count];
    mad.lo.u32 %r_idx, %r_cta_id, %r_ntid, %r_tid;
    mul.lo.u32 %r_stride, %r_nctas, %r_ntid;

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
    @%p_is_worker bra BLOCK_BARRIER_SYNC;

    atom.acq_rel.sys.global.add.u32 %r_prev, [%rd_sync], 1;
    add.u32 %r_arrived, %r_prev, 1;
    setp.eq.u32 %p_last, %r_arrived, %r_nctas;
    @%p_last bra LAST_BLOCK;

WAIT_LAST_BLOCK:
    ld.acquire.sys.global.u32 %r_done_id, [%rd_sync + 4];
    setp.ne.u32 %p_wait, %r_done_id, %expected_task_id;
    @%p_wait bra WAIT_LAST_BLOCK;
    bra BLOCK_BARRIER_SYNC;

LAST_BLOCK:
    st.global.u32 [%rd_sync], 0;
    st.release.sys.global.u32 [%rd_sync + 4], %expected_task_id;
    mov.u32 %status, 1;
    st.release.sys.global.u32 [%rd_slot_ptr + 12], %status;

BLOCK_BARRIER_SYNC:
    bar.sync 0;
    add.u32 %expected_task_id, %expected_task_id, 1;
    add.u32 %r_slot, %r_slot, 1;
    and.b32 %r_slot, %r_slot, %r_mask;
    bra POLL_LOOP;

EXIT_WORKER:
    bar.sync 0;
    @%p_is_worker bra DONE_EXIT;

    atom.acq_rel.sys.global.add.u32 %r_prev, [%rd_sync], 1;
    add.u32 %r_arrived, %r_prev, 1;
    setp.eq.u32 %p_last, %r_arrived, %r_nctas;
    @%p_last bra LAST_EXIT;

WAIT_LAST_EXIT:
    ld.acquire.sys.global.u32 %r_done_id, [%rd_sync + 4];
    setp.ne.u32 %p_wait, %r_done_id, %expected_task_id;
    @%p_wait bra WAIT_LAST_EXIT;
    bra DONE_EXIT;

LAST_EXIT:
    st.global.u32 [%rd_sync], 0;
    st.release.sys.global.u32 [%rd_sync + 4], %expected_task_id;
    mov.u32 %status, 1;
    st.release.sys.global.u32 [%rd_slot_ptr + 12], %status;

DONE_EXIT:
    bar.sync 0;
    ret;
}
";

    /// <summary>
    /// Fused in-VRAM argmax and temperature-scaled softmax reduction kernel.
    /// Emits 8-byte ClavierDecision (uint ActionId, float Confidence) directly in GPU device memory.
    /// </summary>
    public const string ArgmaxReductionPtx = @"
.version 7.5
.target sm_75
.address_size 64

.visible .entry argmax_reduction_f32(
    .param .u64 d_logits,
    .param .u64 d_decision,
    .param .u32 k_classes,
    .param .f32 temperature
)
{
    .shared .align 4 .f32 s_val[1024];
    .shared .align 4 .u32 s_idx[1024];

    .reg .pred %p_empty, %p_t0, %p_bad_t, %p_done, %p_gt, %p_first, %p_take, %p_fold, %p_other_valid, %p_my_empty, %p_eq, %p_lt_id, %p_tie_win, %p_val_win, %p_win, %p_oob_tid;
    .reg .b32 %r_tid, %nthreads, %k, %i, %best_id, %local_id, %active, %half, %limit, %other_tid, %my_id, %other_id;
    .reg .b64 %rd_logits, %rd_decision, %rd_offset, %rd_ptr, %rd_s_val_base, %rd_s_idx_base, %rd_s_val, %rd_s_idx, %rd_s_other_off, %rd_s_other_val, %rd_s_other_idx;
    .reg .f32 %temp, %inv_t, %logit, %scaled, %local_max, %max_val, %my_val, %other_val, %local_sum, %sum_exp, %exp_val, %conf;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %nthreads, %ntid.x;
    min.u32 %nthreads, %nthreads, 1024;

    ld.param.u64 %rd_logits, [d_logits];
    ld.param.u64 %rd_decision, [d_decision];
    ld.param.u32 %k, [k_classes];
    ld.param.f32 %temp, [temperature];

    setp.eq.u32 %p_empty, %k, 0;
    @%p_empty bra HANDLE_EMPTY;

    setp.ge.u32 %p_oob_tid, %r_tid, 1024;
    @%p_oob_tid bra DONE;

    setp.leu.f32 %p_bad_t, %temp, 0f00000000;
    @%p_bad_t mov.f32 %temp, 0f3F800000;
    rcp.approx.f32 %inv_t, %temp;

    mov.f32 %local_max, 0fFF800000;
    mov.u32 %local_id, 0xFFFFFFFF;
    mov.u32 %i, %r_tid;

STRIDE_MAX_LOOP:
    setp.ge.u32 %p_done, %i, %k;
    @%p_done bra STRIDE_MAX_DONE;

    cvt.u64.u32 %rd_offset, %i;
    shl.b64 %rd_offset, %rd_offset, 2;
    add.s64 %rd_ptr, %rd_logits, %rd_offset;
    ld.global.f32 %logit, [%rd_ptr];
    mul.f32 %scaled, %logit, %inv_t;

    setp.gt.f32 %p_gt, %scaled, %local_max;
    setp.eq.u32 %p_first, %local_id, 0xFFFFFFFF;
    or.pred %p_take, %p_gt, %p_first;
    @%p_take mov.f32 %local_max, %scaled;
    @%p_take mov.u32 %local_id, %i;

    add.u32 %i, %i, %nthreads;
    bra STRIDE_MAX_LOOP;

STRIDE_MAX_DONE:
    cvt.u64.u32 %rd_offset, %r_tid;
    shl.b64 %rd_offset, %rd_offset, 2;
    mov.u64 %rd_s_val_base, s_val;
    mov.u64 %rd_s_idx_base, s_idx;
    add.u64 %rd_s_val, %rd_s_val_base, %rd_offset;
    add.u64 %rd_s_idx, %rd_s_idx_base, %rd_offset;
    st.shared.f32 [%rd_s_val], %local_max;
    st.shared.u32 [%rd_s_idx], %local_id;
    bar.sync 0;

    mov.u32 %active, %nthreads;

REDUCE_MAX_LOOP:
    setp.le.u32 %p_done, %active, 1;
    @%p_done bra REDUCE_MAX_DONE;

    add.u32 %half, %active, 1;
    shr.u32 %half, %half, 1;
    shr.u32 %limit, %active, 1;
    setp.lt.u32 %p_fold, %r_tid, %limit;
    @!%p_fold bra REDUCE_MAX_SYNC;

    add.u32 %other_tid, %r_tid, %half;
    cvt.u64.u32 %rd_s_other_off, %other_tid;
    shl.b64 %rd_s_other_off, %rd_s_other_off, 2;
    add.u64 %rd_s_other_val, %rd_s_val_base, %rd_s_other_off;
    add.u64 %rd_s_other_idx, %rd_s_idx_base, %rd_s_other_off;

    ld.shared.f32 %my_val, [%rd_s_val];
    ld.shared.u32 %my_id, [%rd_s_idx];
    ld.shared.f32 %other_val, [%rd_s_other_val];
    ld.shared.u32 %other_id, [%rd_s_other_idx];

    setp.ne.u32 %p_other_valid, %other_id, 0xFFFFFFFF;
    setp.eq.u32 %p_my_empty, %my_id, 0xFFFFFFFF;
    setp.gt.f32 %p_gt, %other_val, %my_val;
    setp.eq.f32 %p_eq, %other_val, %my_val;
    setp.lt.u32 %p_lt_id, %other_id, %my_id;
    and.pred %p_tie_win, %p_eq, %p_lt_id;
    or.pred %p_val_win, %p_gt, %p_tie_win;
    or.pred %p_win, %p_val_win, %p_my_empty;
    and.pred %p_win, %p_win, %p_other_valid;

    @%p_win st.shared.f32 [%rd_s_val], %other_val;
    @%p_win st.shared.u32 [%rd_s_idx], %other_id;

REDUCE_MAX_SYNC:
    bar.sync 0;
    mov.u32 %active, %half;
    bra REDUCE_MAX_LOOP;

REDUCE_MAX_DONE:
    ld.shared.f32 %max_val, [s_val];
    ld.shared.u32 %best_id, [s_idx];

    mov.f32 %local_sum, 0f00000000;
    mov.u32 %i, %r_tid;

STRIDE_SUM_LOOP:
    setp.ge.u32 %p_done, %i, %k;
    @%p_done bra STRIDE_SUM_DONE;

    cvt.u64.u32 %rd_offset, %i;
    shl.b64 %rd_offset, %rd_offset, 2;
    add.s64 %rd_ptr, %rd_logits, %rd_offset;
    ld.global.f32 %logit, [%rd_ptr];
    mul.f32 %scaled, %logit, %inv_t;
    sub.f32 %scaled, %scaled, %max_val;
    mul.f32 %scaled, %scaled, 0f3FB8AA3B;
    ex2.approx.f32 %exp_val, %scaled;
    add.f32 %local_sum, %local_sum, %exp_val;

    add.u32 %i, %i, %nthreads;
    bra STRIDE_SUM_LOOP;

STRIDE_SUM_DONE:
    bar.sync 0;
    st.shared.f32 [%rd_s_val], %local_sum;
    bar.sync 0;

    mov.u32 %active, %nthreads;

REDUCE_SUM_LOOP:
    setp.le.u32 %p_done, %active, 1;
    @%p_done bra REDUCE_SUM_DONE;

    add.u32 %half, %active, 1;
    shr.u32 %half, %half, 1;
    shr.u32 %limit, %active, 1;
    setp.lt.u32 %p_fold, %r_tid, %limit;
    @!%p_fold bra REDUCE_SUM_SYNC;

    add.u32 %other_tid, %r_tid, %half;
    cvt.u64.u32 %rd_s_other_off, %other_tid;
    shl.b64 %rd_s_other_off, %rd_s_other_off, 2;
    add.u64 %rd_s_other_val, %rd_s_val_base, %rd_s_other_off;

    ld.shared.f32 %my_val, [%rd_s_val];
    ld.shared.f32 %other_val, [%rd_s_other_val];
    add.f32 %my_val, %my_val, %other_val;
    st.shared.f32 [%rd_s_val], %my_val;

REDUCE_SUM_SYNC:
    bar.sync 0;
    mov.u32 %active, %half;
    bra REDUCE_SUM_LOOP;

REDUCE_SUM_DONE:
    setp.eq.u32 %p_t0, %r_tid, 0;
    @!%p_t0 bra DONE;

    ld.shared.f32 %sum_exp, [s_val];
    rcp.approx.f32 %conf, %sum_exp;

    st.global.u32 [%rd_decision], %best_id;
    st.global.f32 [%rd_decision + 4], %conf;
    bra DONE;

HANDLE_EMPTY:
    setp.eq.u32 %p_t0, %r_tid, 0;
    @!%p_t0 bra DONE;

    mov.u32 %best_id, 0;
    mov.f32 %conf, 0f00000000;
    st.global.u32 [%rd_decision], %best_id;
    st.global.f32 [%rd_decision + 4], %conf;

DONE:
    ret;
}
";
}
