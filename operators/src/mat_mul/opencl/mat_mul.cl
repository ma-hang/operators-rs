#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef Tval
#define Tval float
#endif

#ifdef USE_HALF
#define MUL(valueA, valueB) (float) (valueA * valueB)
#define SCAL(beta, p, alpha, value) (half)(beta * (float) (*p) + alpha * value)
#else
#define MUL(valueA, valueB) valueA *valueB
#define SCAL(beta, p, alpha, value) beta *(*p) + alpha *value
#endif

__kernel void general_gemm(__global Tval *A, __global Tval *B, __global Tval *C,
                           int as, int ars, int acs, int bs, int brs, int bcs,
                           int cs, int crs, int ccs, int batch,
                           int M, int N, int K, float alpha, float beta) {
    int g_idx = get_global_id(1);
    int g_idy = get_global_id(0);
    int row_id = g_idy / N;
    int col_id = g_idy % N;

    Tval valueA = 0.0f;
    Tval valueB = 0.0f;
    float value = 0.0f;

    for (int i = 0; i < K; i++) {
        valueA = *(A + g_idx * as + row_id * ars + i * acs);
        valueB = *(B + g_idx * bs + i * brs + col_id * bcs);
        value += MUL(valueA, valueB);
    }

    __global Tval *p = C + g_idx * cs + row_id * crs + col_id * ccs;
    *p = SCAL(beta, p, alpha, value);
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void
gemm_f16_v1(__global half *A, __global half *B, __global half *C,
            uint as, uint ars, uint acs, uint bs, uint brs, uint bcs,
            uint cs, uint crs, uint ccs, uint batch,
            uint M, uint N, uint K, float alpha, float beta) {

    uint batch_id = get_global_id(1);
    uint group_id = get_group_id(0);                  // 工作组编号
    uint num_subgroups = get_num_sub_groups();        // 当前工作组中的子组数
    uint subgroup_id = get_sub_group_id();            // 当前子组在工作组内的编号
    uint subgroup_local_id = get_sub_group_local_id();// 子组内工作项的索引
    uint subgroup_size = get_sub_group_size();        // 子组大小   ---32

    uint row = group_id * num_subgroups + subgroup_id;
    if (row >= M)
        return;

    float partial_sum = 0.0f;
    for (uint col = subgroup_local_id; col < K; col += subgroup_size) {
        partial_sum += (float) (A[batch_id * as + row * ars + col * acs] * B[batch_id * bs + col * brs]);
    }

    float sum = sub_group_reduce_add(partial_sum);

    if (subgroup_local_id == 0) {
        C[batch_id * cs + row * crs] = (half) (alpha * sum + beta * (float) C[batch_id * cs + row * crs]);
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void
gemm_f16_v2(__global half *A, __global half *B, __global half *C,
            uint as, uint ars, uint acs, uint bs, uint brs, uint bcs,
            uint cs, uint crs, uint ccs, uint batch,
            uint M, uint N, uint K, float alpha, float beta) {

    uint batch_id = get_global_id(1);
    uint group_id = get_group_id(0);                  // 工作组编号
    uint num_subgroups = get_num_sub_groups();        // 当前工作组中的子组数
    uint subgroup_id = get_sub_group_id();            // 当前子组在工作组内的编号
    uint subgroup_local_id = get_sub_group_local_id();// 子组内工作项的索引
    uint subgroup_size = get_sub_group_size() * 8;    // 子组大小   ---32

    uint row = group_id * num_subgroups + subgroup_id;
    if (row >= M)
        return;

    half8 aa;
    half8 bb;
    half8 cc;
    half8 partial_sum = 0;
    for (uint col = subgroup_local_id * 8; col < K; col += subgroup_size) {

        aa = vload8(0, A + batch_id * as + row * ars + col * acs);
        bb = vload8(0, B + batch_id * bs + col * brs);
        cc = aa * bb;
        partial_sum += cc;
    }

    float sumbase = (float) (partial_sum.s0 + partial_sum.s1 + partial_sum.s2 + partial_sum.s3 + partial_sum.s4 + partial_sum.s5 + partial_sum.s6 + partial_sum.s7);
    float sum = sub_group_reduce_add(sumbase);

    if (subgroup_local_id == 0) {
        C[batch_id * cs + row * crs] = (half) (alpha * sum + beta * (float) C[batch_id * cs + row * crs]);
    }
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void
gemm_f16_v3(__global half *A, __global half *B, __global half *C,
            uint as, uint ars, uint acs, uint bs, uint brs, uint bcs,
            uint cs, uint crs, uint ccs, uint batch,
            uint M, uint N, uint K, float alpha, float beta) {

    uint batch_id = get_global_id(1);
    uint group_id = get_group_id(0);                  // 工作组编号
    uint num_subgroups = get_num_sub_groups();        // 当前工作组中的子组数
    uint subgroup_id = get_sub_group_id();            // 当前子组在工作组内的编号
    uint subgroup_local_id = get_sub_group_local_id();// 子组内工作项的索引
    uint subgroup_size = get_sub_group_size() * 8;    // 子组大小   ---32

    uint row = group_id * num_subgroups + subgroup_id;
    if (row >= M)
        return;

    half8 aa;
    half8 bb;
    half8 cc;
    half8 partial_sum = 0;
    for (uint col = subgroup_local_id * 8; col < K; col += subgroup_size) {

        aa = vload8(0, A + batch_id * as + row * ars + col * acs);
        bb = vload8(0, B + batch_id * bs + col * brs);
        cc = aa * bb;
        partial_sum += cc;
    }

    float sumbase = (float) (partial_sum.s0 + partial_sum.s1 + partial_sum.s2 + partial_sum.s3 + partial_sum.s4 + partial_sum.s5 + partial_sum.s6 + partial_sum.s7);
    float sum = sub_group_reduce_add(sumbase);

    if (subgroup_local_id == 0) {
        C[batch_id * cs + row * crs] = (half) (alpha * sum + beta * (float) C[batch_id * cs + row * crs]);
    }
}


__attribute__((intel_reqd_sub_group_size(32)))
__kernel void
gemm_f16_v4(__global half *A, __global half *B, __global half *C,
            uint as, uint ars, uint acs, uint bs, uint brs, uint bcs,
            uint cs, uint crs, uint ccs, uint batch,
            uint M, uint N, uint K, float alpha, float beta) {

    uint batch_id = get_global_id(1);
    uint group_id = get_group_id(0);                  // 工作组编号
    uint num_subgroups = get_num_sub_groups();        // 当前工作组中的子组数
    uint subgroup_id = get_sub_group_id();            // 当前子组在工作组内的编号
    uint subgroup_local_id = get_sub_group_local_id();// 子组内工作项的索引
    uint subgroup_size = get_sub_group_size() * 8;    // 子组大小   ---32

    uint row = group_id * num_subgroups + subgroup_id;
    if (row >= M)
        return;

    half8 aa;
    half8 bb;
    half8 partial_sum = 0;

    for (uint col = subgroup_local_id * 8; col < K; col += subgroup_size * 4) {

        aa = vload8(0, A + batch_id * as + row * ars + col * acs);
        bb = vload8(0, B + batch_id * bs + col * brs);
        partial_sum += aa * bb;

        aa = vload8(0, A + batch_id * as + row * ars + (col + subgroup_size) * acs);
        bb = vload8(0, B + batch_id * bs + (col + subgroup_size) * brs);
        partial_sum += aa * bb;

        aa = vload8(0, A + batch_id * as + row * ars + (col + subgroup_size * 2) * acs);
        bb = vload8(0, B + batch_id * bs + (col + subgroup_size * 2) * brs);
        partial_sum += aa * bb;

        aa = vload8(0, A + batch_id * as + row * ars + (col + subgroup_size * 3) * acs);
        bb = vload8(0, B + batch_id * bs + (col + subgroup_size * 3) * brs);
        partial_sum += aa * bb;
    }

    float sumbase = (float) (partial_sum.s0 + partial_sum.s1 + partial_sum.s2 + partial_sum.s3 + partial_sum.s4 + partial_sum.s5 + partial_sum.s6 + partial_sum.s7);
    float sum = sub_group_reduce_add(sumbase);

    if (subgroup_local_id == 0) {
        C[batch_id * cs + row * crs] = (half) (alpha * sum + beta * (float) C[batch_id * cs + row * crs]);
    }
}
