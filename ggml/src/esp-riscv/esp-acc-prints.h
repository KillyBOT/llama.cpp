#pragma once

#include <stdio.h>

#include "esp.h"
#include "esp-gemm-cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

void print_gemm_cfg(esp_thread_info_t* thread_cfg, struct gemm_stratus_access* cfg);

void print_gemm_cfg(esp_thread_info_t* thread_cfg, struct gemm_stratus_access* cfg) {
  fprintf(stderr, "---------------- gemm ACC Config ----------------\n");
  fprintf(stderr, "    devname =            %s\n", thread_cfg->devname);
  fprintf(stderr, "    do_relu =            %d\n", cfg->do_relu);
  fprintf(stderr, "    transpose =          %d\n", cfg->transpose);
  fprintf(stderr, "    ninputs =            %d\n", cfg->ninputs);
  fprintf(stderr, "    d3 =                 %d\n", cfg->d3);
  fprintf(stderr, "    d2 =                 %d\n", cfg->d2);
  fprintf(stderr, "    d1 =                 %d\n", cfg->d1);
  fprintf(stderr, "    st_offset =          %d\n", cfg->st_offset);
  fprintf(stderr, "    ld_offset1 =         %d\n", cfg->ld_offset1);
  fprintf(stderr, "    ld_offset2 =         %d\n", cfg->ld_offset2);
}

#ifdef __cplusplus
}
#endif
