#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <linux/ioctl.h>

#include "esp.h"

struct gemm_stratus_access {
	struct esp_access esp;
	/* <<--regs-->> */
	unsigned do_relu;
	unsigned transpose;
	unsigned ninputs;
	unsigned d3;
	unsigned d2;
	unsigned d1;
	unsigned st_offset;
	unsigned ld_offset1;
	unsigned ld_offset2;
	unsigned src_offset;
	unsigned dst_offset;
};

/* We are using fixed point */
typedef int esp_token_t;
typedef float esp_native_t;
#define fx2float i32_to_f32
#define float2fx f32_to_i32
#define FX_IL 16

#define MAX_PRINTED_ERRORS 512

// Only one accumulator for now...
#define NACC 1
#define ACC_TLB_ENTRIES 128
#define ACC_PAGE_SIZE (1 << 20)
#define MAX_SIZE (ACC_PAGE_SIZE * ACC_TLB_ENTRIES)

struct gemm_stratus_access gemm_cfg_000[] = {
	{
		.esp.contig = NULL,
        .esp.run = 0,
        .esp.p2p_store = 0,
        .esp.p2p_nsrcs = 0,
        .esp.p2p_srcs = {"", "", "", ""},
        .esp.coherence = ACC_COH_NONE,
        .esp.footprint = 0,
        .esp.alloc_policy = CONTIG_ALLOC_PREFERRED,
        .esp.ddr_node = 0,
        .esp.in_place = 0,
        .esp.reuse_factor = 0,
		/* <<--descriptor-->> */
		.do_relu = 0,
		.transpose = 1,
		.ninputs = 2,
		.d3 = 8,
		.d2 = 8,
		.d1 = 8,
		.st_offset = 256,
		.ld_offset1 = 0,
		.ld_offset2 = 128,
		.src_offset = 0,
		.dst_offset = 0,
	}
};

#define GEMM_STRATUS_IOC_ACCESS _IOW('S', 0, struct gemm_stratus_access)

esp_thread_info_t thread_cfg_000[] = {
	{
		.run = true,
		.devname = "gemm_stratus.0",
		.hw_buf = NULL,
		.ioctl_req = GEMM_STRATUS_IOC_ACCESS,
		.esp_desc = &(gemm_cfg_000[0].esp),
		.fd = 0,
		.hw_ns = 0,
	}
};
