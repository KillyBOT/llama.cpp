#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// esp_accelerator.h
enum accelerator_coherence {ACC_COH_NONE = 0, ACC_COH_LLC, ACC_COH_RECALL, ACC_COH_FULL, ACC_COH_AUTO};

// contig_alloc.h
struct contig_khandle_struct {
	char unused;
};
typedef struct contig_khandle_struct *contig_khandle_t;

enum contig_alloc_policy {
	CONTIG_ALLOC_PREFERRED,
	CONTIG_ALLOC_LEAST_LOADED,
	CONTIG_ALLOC_BALANCED,
};

// esp.h
struct esp_access {
	contig_khandle_t contig;
	uint8_t run;
	uint8_t p2p_store;
	uint8_t p2p_nsrcs;
	char p2p_srcs[4][64];
	enum accelerator_coherence coherence;
    unsigned int footprint;
    enum contig_alloc_policy alloc_policy;
    unsigned int ddr_node;
	unsigned int in_place;
	unsigned int reuse_factor;
};

// libesp.h

typedef struct esp_accelerator_thread_info {
	bool run;
	char *devname;
	void *hw_buf;
	int ioctl_req;
	/* Partially Filled-in by ESPLIB */
	struct esp_access *esp_desc;
	/* Filled-in by ESPLIB */
	int fd;
	unsigned long long hw_ns;
} esp_thread_info_t;

#ifdef __cplusplus
extern "C" {
#endif

void *esp_alloc(size_t size);
void esp_run(esp_thread_info_t cfg[], unsigned nacc);
unsigned long long esp_run_no_print(esp_thread_info_t cfg[], unsigned nacc);
void esp_free(void *buf);

#ifdef __cplusplus
}
#endif
