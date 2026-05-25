/*
 * tensorcore - minimal GGUF v3 reader.
 *
 * Memory-mapped: the file is mmap'd read-only and tensor info structures
 * point into the mapping (zero-copy). Metadata keys are stored in a flat
 * array; lookups are linear (model files have ~few hundred KV pairs at most).
 *
 * Spec reference: github.com/ggml-org/ggml/blob/master/docs/gguf.md
 *
 * Type IDs follow GGML enum (ggml.h):
 *   F32=0, F16=1, Q4_0=2, Q4_1=3, Q8_0=8, ... BF16=30
 */

#include "tensorcore/gguf.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747u   /* 'GGUF' little-endian */
#define GGUF_DEFAULT_ALIGNMENT 32

typedef enum {
    GGUF_TYPE_UINT8   = 0, GGUF_TYPE_INT8   = 1,
    GGUF_TYPE_UINT16  = 2, GGUF_TYPE_INT16  = 3,
    GGUF_TYPE_UINT32  = 4, GGUF_TYPE_INT32  = 5,
    GGUF_TYPE_FLOAT32 = 6, GGUF_TYPE_BOOL   = 7,
    GGUF_TYPE_STRING  = 8, GGUF_TYPE_ARRAY  = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type_t;

typedef struct {
    char*             key;            /* heap-owned copy */
    gguf_value_type_t type;
    /* For scalar types we store raw bytes; strings are copied and arrays carry
     * pointers into the mmap. */
    union {
        uint8_t  u8;  int8_t  i8;
        uint16_t u16; int16_t i16;
        uint32_t u32; int32_t i32;
        uint64_t u64; int64_t i64;
        float    f32; double  f64;
        int      boolean;
        struct { char* p; uint64_t n; } str;  /* heap-owned NUL-terminated copy */
        struct { gguf_value_type_t elem; const void* p; uint64_t n; } arr;
    } v;
} gguf_kv;

struct tc_gguf_file {
#if defined(_WIN32)
    HANDLE    file_handle;
    HANDLE    mapping_handle;
#else
    int       fd;
#endif
    void*     map;
    size_t    map_size;
    /* Header. */
    uint32_t  version;
    uint64_t  tensor_count;
    uint64_t  metadata_kv_count;
    /* Metadata. */
    gguf_kv*  kvs;
    /* Tensor info entries: heap array with tensor data pointers into the mmap. */
    tc_gguf_tensor_info* tensors;
    /* Where tensor data starts (after alignment padding). */
    uint64_t  data_offset;
    uint32_t  alignment;
};

typedef struct {
    char*            name;
    int32_t          n_dims;
    uint64_t         dims[4];
    tc_gguf_type_t   type;
    uint64_t         offset;
    size_t           n_bytes;
    tc_buffer*       buffer;
} loaded_tensor;

struct tc_gguf_loaded_model {
    uint64_t       count;
    uint64_t       skipped;
    loaded_tensor* tensors;
};

static tc_status_t tc_gguf_tensor_info_to_buffer(tc_context* ctx,
                                                 const tc_gguf_tensor_info* info,
                                                 tc_buffer** out_buffer);
static tc_status_t gguf_quantized_matrix_info_common(int32_t n_dims,
                                                     const uint64_t dims[4],
                                                     tc_gguf_type_t type,
                                                     size_t n_bytes,
                                                     tc_buffer* buffer,
                                                     tc_gguf_quantized_matrix_info* out_info);

static char* gguf_strdup(const char* src) {
    if (!src) src = "";
    const size_t n = strlen(src);
    char* dst = (char*)malloc(n + 1);
    if (!dst) return NULL;
    memcpy(dst, src, n + 1);
    return dst;
}

#if defined(_WIN32)
static tc_status_t map_file_readonly(const char* path,
                                     void** out_map,
                                     size_t* out_size,
                                     HANDLE* out_file,
                                     HANDLE* out_mapping) {
    if (!out_map || !out_size || !out_file || !out_mapping) return TC_ERR_INVALID_ARG;
    *out_map = NULL;
    *out_size = 0;
    *out_file = INVALID_HANDLE_VALUE;
    *out_mapping = NULL;

    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return TC_ERR_INTERNAL;

    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0 ||
        (uint64_t)size.QuadPart > (uint64_t)SIZE_MAX) {
        CloseHandle(file);
        return TC_ERR_INTERNAL;
    }

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) {
        CloseHandle(file);
        return TC_ERR_INTERNAL;
    }

    void* map = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!map) {
        CloseHandle(mapping);
        CloseHandle(file);
        return TC_ERR_INTERNAL;
    }

    *out_map = map;
    *out_size = (size_t)size.QuadPart;
    *out_file = file;
    *out_mapping = mapping;
    return TC_OK;
}
#else
static tc_status_t map_file_readonly(const char* path,
                                     void** out_map,
                                     size_t* out_size,
                                     int* out_fd) {
    if (!out_map || !out_size || !out_fd) return TC_ERR_INVALID_ARG;
    *out_map = NULL;
    *out_size = 0;
    *out_fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return TC_ERR_INTERNAL;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return TC_ERR_INTERNAL; }
    if (st.st_size <= 0 || (uint64_t)st.st_size > SIZE_MAX) {
        close(fd);
        return TC_ERR_INTERNAL;
    }
    void* map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { close(fd); return TC_ERR_INTERNAL; }

    *out_map = map;
    *out_size = (size_t)st.st_size;
    *out_fd = fd;
    return TC_OK;
}
#endif

/* ----------------- Read helpers ----------------- */
typedef struct {
    const uint8_t* base;
    size_t         pos;
    size_t         size;
} reader_t;

static int rd_bytes(reader_t* r, void* dst, size_t n) {
    if (r->pos > r->size || n > r->size - r->pos) return -1;
    memcpy(dst, r->base + r->pos, n);
    r->pos += n;
    return 0;
}
static int rd_u32(reader_t* r, uint32_t* v) { return rd_bytes(r, v, 4); }
static int rd_u64(reader_t* r, uint64_t* v) { return rd_bytes(r, v, 8); }
static int rd_str(reader_t* r, const char** out_ptr, uint64_t* out_n) {
    uint64_t n;
    if (rd_u64(r, &n) != 0) return -1;
    if (r->pos > r->size || n > r->size - r->pos) return -1;
    *out_ptr = (const char*)(r->base + r->pos);
    *out_n = n;
    r->pos += n;
    return 0;
}

/* Read a string into a heap-owned NUL-terminated buffer. */
static char* rd_str_dup_n(reader_t* r, uint64_t* out_n) {
    const char* p; uint64_t n;
    if (rd_str(r, &p, &n) != 0) return NULL;
    if (n > SIZE_MAX - 1) return NULL;
    char* s = (char*)malloc(n + 1);
    if (!s) return NULL;
    memcpy(s, p, n);
    s[n] = '\0';
    if (out_n) *out_n = n;
    return s;
}

static char* rd_str_dup(reader_t* r) {
    return rd_str_dup_n(r, NULL);
}

static size_t gguf_scalar_size(gguf_value_type_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static int rd_value(reader_t* r, gguf_value_type_t type, gguf_kv* kv) {
    kv->type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:   return rd_bytes(r, &kv->v.u8,  1);
        case GGUF_TYPE_INT8:    return rd_bytes(r, &kv->v.i8,  1);
        case GGUF_TYPE_UINT16:  return rd_bytes(r, &kv->v.u16, 2);
        case GGUF_TYPE_INT16:   return rd_bytes(r, &kv->v.i16, 2);
        case GGUF_TYPE_UINT32:  return rd_bytes(r, &kv->v.u32, 4);
        case GGUF_TYPE_INT32:   return rd_bytes(r, &kv->v.i32, 4);
        case GGUF_TYPE_FLOAT32: return rd_bytes(r, &kv->v.f32, 4);
        case GGUF_TYPE_BOOL: {
            uint8_t b; if (rd_bytes(r, &b, 1) != 0) return -1;
            kv->v.boolean = b; return 0;
        }
        case GGUF_TYPE_STRING:
            kv->v.str.p = rd_str_dup_n(r, &kv->v.str.n);
            return kv->v.str.p ? 0 : -1;
        case GGUF_TYPE_UINT64:  return rd_bytes(r, &kv->v.u64, 8);
        case GGUF_TYPE_INT64:   return rd_bytes(r, &kv->v.i64, 8);
        case GGUF_TYPE_FLOAT64: return rd_bytes(r, &kv->v.f64, 8);
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_t; uint64_t n;
            if (rd_u32(r, &elem_t) != 0) return -1;
            if (rd_u64(r, &n)      != 0) return -1;
            kv->v.arr.elem = (gguf_value_type_t)elem_t;
            kv->v.arr.p    = r->base + r->pos;
            kv->v.arr.n    = n;
            /* Skip past the array data. For variable-length elements (strings)
             * we have to walk each entry to compute the size. */
            if (elem_t == GGUF_TYPE_STRING) {
                for (uint64_t i = 0; i < n; ++i) {
                    const char* p; uint64_t sn;
                    if (rd_str(r, &p, &sn) != 0) return -1;
                }
            } else {
                const size_t elem_size = gguf_scalar_size((gguf_value_type_t)elem_t);
                if (elem_size == 0) return -1;
                if (r->pos > r->size || n > (r->size - r->pos) / elem_size) return -1;
                r->pos += n * elem_size;
            }
            return 0;
        }
    }
    return -1;
}

/* Map GGML ggml_type enum value to tensorcore type. */
static tc_gguf_type_t map_ggml_type(uint32_t t) {
    switch (t) {
        case 0: return TC_GGUF_TYPE_F32;
        case 1: return TC_GGUF_TYPE_F16;
        case 2: return TC_GGUF_TYPE_Q4_0;
        case 3: return TC_GGUF_TYPE_Q4_1;
        case 8: return TC_GGUF_TYPE_Q8_0;
        case 30: return TC_GGUF_TYPE_BF16;
        default: return TC_GGUF_TYPE_UNSUPPORTED;
    }
}

/* Size-in-bytes computation. Hardcoded for the types we care about. */
static int type_bytes(tc_gguf_type_t t, uint64_t n_elems, size_t* out_bytes) {
    if (!out_bytes) return -1;
    *out_bytes = 0;
    switch (t) {
        case TC_GGUF_TYPE_F32:
            if (n_elems > SIZE_MAX / 4) return -1;
            *out_bytes = (size_t)n_elems * 4;
            return 0;
        case TC_GGUF_TYPE_F16:
        case TC_GGUF_TYPE_BF16:
            if (n_elems > SIZE_MAX / 2) return -1;
            *out_bytes = (size_t)n_elems * 2;
            return 0;
        case TC_GGUF_TYPE_Q4_0:
            if (n_elems % 32 != 0 || n_elems / 32 > SIZE_MAX / 18) return -1;
            *out_bytes = (size_t)(n_elems / 32) * 18;
            return 0;
        case TC_GGUF_TYPE_Q4_1:
            if (n_elems % 32 != 0 || n_elems / 32 > SIZE_MAX / 20) return -1;
            *out_bytes = (size_t)(n_elems / 32) * 20;
            return 0;
        case TC_GGUF_TYPE_Q8_0:
            if (n_elems % 32 != 0 || n_elems / 32 > SIZE_MAX / 34) return -1;
            *out_bytes = (size_t)(n_elems / 32) * 34;
            return 0;
        default:
            return 0;
    }
}

tc_status_t tc_gguf_open(const char* path, tc_gguf_file** out) {
    if (!path || !out) return TC_ERR_INVALID_ARG;
    void* map = NULL;
    size_t map_size = 0;
#if defined(_WIN32)
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = NULL;
    tc_status_t map_status = map_file_readonly(path, &map, &map_size,
                                               &file_handle, &mapping_handle);
#else
    int fd = -1;
    tc_status_t map_status = map_file_readonly(path, &map, &map_size, &fd);
#endif
    if (map_status != TC_OK) return map_status;

    tc_gguf_file* f = (tc_gguf_file*)calloc(1, sizeof(*f));
    if (!f) {
#if defined(_WIN32)
        if (map) UnmapViewOfFile(map);
        if (mapping_handle) CloseHandle(mapping_handle);
        if (file_handle != INVALID_HANDLE_VALUE) CloseHandle(file_handle);
#else
        munmap(map, map_size);
        close(fd);
#endif
        return TC_ERR_ALLOC;
    }
#if defined(_WIN32)
    f->file_handle = file_handle;
    f->mapping_handle = mapping_handle;
#else
    f->fd = fd;
#endif
    f->map = map;
    f->map_size = map_size;
    f->alignment = GGUF_DEFAULT_ALIGNMENT;

    reader_t r = { (const uint8_t*)map, 0, map_size };

    uint32_t magic;
    if (rd_u32(&r, &magic) != 0 || magic != GGUF_MAGIC) goto fail;
    if (rd_u32(&r, &f->version) != 0)        goto fail;
    if (f->version != 3)                     goto fail;
    if (rd_u64(&r, &f->tensor_count) != 0)   goto fail;
    if (rd_u64(&r, &f->metadata_kv_count) != 0) goto fail;

    if (f->metadata_kv_count > SIZE_MAX / sizeof(gguf_kv)) goto fail;
    f->kvs = (gguf_kv*)calloc(f->metadata_kv_count, sizeof(gguf_kv));
    if (f->metadata_kv_count && !f->kvs) goto fail;
    for (uint64_t i = 0; i < f->metadata_kv_count; ++i) {
        f->kvs[i].key = rd_str_dup(&r);
        if (!f->kvs[i].key) goto fail;
        uint32_t val_t;
        if (rd_u32(&r, &val_t) != 0) goto fail;
        if (rd_value(&r, (gguf_value_type_t)val_t, &f->kvs[i]) != 0) goto fail;
        /* general.alignment override. */
        if (strcmp(f->kvs[i].key, "general.alignment") == 0 &&
            f->kvs[i].type == GGUF_TYPE_UINT32) {
            f->alignment = f->kvs[i].v.u32;
        }
    }
    if (f->alignment == 0) goto fail;

    if (f->tensor_count > SIZE_MAX / sizeof(tc_gguf_tensor_info)) goto fail;
    f->tensors = (tc_gguf_tensor_info*)calloc(f->tensor_count, sizeof(tc_gguf_tensor_info));
    if (f->tensor_count && !f->tensors) goto fail;
    for (uint64_t i = 0; i < f->tensor_count; ++i) {
        const char* name_p; uint64_t name_n;
        if (rd_str(&r, &name_p, &name_n) != 0) goto fail;
        /* Make NUL-terminated copy. */
        if (name_n > SIZE_MAX - 1) goto fail;
        char* nbuf = (char*)malloc(name_n + 1);
        if (!nbuf) goto fail;
        memcpy(nbuf, name_p, name_n);
        nbuf[name_n] = '\0';
        f->tensors[i].name = nbuf;

        uint32_t n_dims;
        if (rd_u32(&r, &n_dims) != 0) goto fail;
        if (n_dims > 4) goto fail;
        f->tensors[i].n_dims = (int32_t)n_dims;
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (rd_u64(&r, &f->tensors[i].dims[d]) != 0) goto fail;
            if (f->tensors[i].dims[d] == 0 ||
                n_elems > UINT64_MAX / f->tensors[i].dims[d]) goto fail;
            n_elems *= f->tensors[i].dims[d];
        }
        uint32_t ggml_t;
        if (rd_u32(&r, &ggml_t) != 0)              goto fail;
        f->tensors[i].type = map_ggml_type(ggml_t);
        if (rd_u64(&r, &f->tensors[i].offset) != 0) goto fail;
        if (type_bytes(f->tensors[i].type, n_elems, &f->tensors[i].n_bytes) != 0)
            goto fail;
    }

    /* Align r.pos up to f->alignment for data start. */
    const uint64_t align = f->alignment;
    const uint64_t pad = (align - (r.pos % align)) % align;
    if (pad > UINT64_MAX - r.pos) goto fail;
    f->data_offset = r.pos + pad;
    if (f->data_offset > f->map_size) goto fail;

    /* Resolve data pointers. */
    for (uint64_t i = 0; i < f->tensor_count; ++i) {
        if (f->tensors[i].offset > UINT64_MAX - f->data_offset ||
            f->data_offset + f->tensors[i].offset > f->map_size) {
            goto fail;
        }
        const uint64_t tensor_start = f->data_offset + f->tensors[i].offset;
        if (f->tensors[i].n_bytes > f->map_size - tensor_start) {
            goto fail;
        }
        f->tensors[i].data = (const uint8_t*)map + f->data_offset + f->tensors[i].offset;
    }

    *out = f;
    return TC_OK;
fail:
    tc_gguf_close(f);
    return TC_ERR_INTERNAL;
}

void tc_gguf_close(tc_gguf_file* f) {
    if (!f) return;
    if (f->tensors) {
        for (uint64_t i = 0; i < f->tensor_count; ++i) free((void*)f->tensors[i].name);
        free(f->tensors);
    }
    if (f->kvs) {
        for (uint64_t i = 0; i < f->metadata_kv_count; ++i) {
            free(f->kvs[i].key);
            if (f->kvs[i].type == GGUF_TYPE_STRING) free(f->kvs[i].v.str.p);
        }
        free(f->kvs);
    }
#if defined(_WIN32)
    if (f->map) UnmapViewOfFile(f->map);
    if (f->mapping_handle) CloseHandle(f->mapping_handle);
    if (f->file_handle && f->file_handle != INVALID_HANDLE_VALUE) CloseHandle(f->file_handle);
#else
    if (f->map && f->map != MAP_FAILED) munmap(f->map, f->map_size);
    if (f->fd >= 0) close(f->fd);
#endif
    free(f);
}

uint64_t tc_gguf_tensor_count(const tc_gguf_file* f) { return f ? f->tensor_count : 0; }
uint64_t tc_gguf_metadata_count(const tc_gguf_file* f) { return f ? f->metadata_kv_count : 0; }

tc_status_t tc_gguf_get_tensor(const tc_gguf_file* f, const char* name,
                               tc_gguf_tensor_info* out_info) {
    if (!f || !name || !out_info) return TC_ERR_INVALID_ARG;
    for (uint64_t i = 0; i < f->tensor_count; ++i) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            *out_info = f->tensors[i];
            return TC_OK;
        }
    }
    return TC_ERR_INVALID_ARG;
}

tc_status_t tc_gguf_tensor_at(const tc_gguf_file* f, uint64_t i,
                              tc_gguf_tensor_info* out_info) {
    if (!f || !out_info || i >= f->tensor_count) return TC_ERR_INVALID_ARG;
    *out_info = f->tensors[i];
    return TC_OK;
}

const char* tc_gguf_meta_get_str(const tc_gguf_file* f, const char* key) {
    if (!f || !key) return NULL;
    for (uint64_t i = 0; i < f->metadata_kv_count; ++i) {
        if (strcmp(f->kvs[i].key, key) == 0 &&
            f->kvs[i].type == GGUF_TYPE_STRING) {
            return f->kvs[i].v.str.p;
        }
    }
    return NULL;
}

int64_t tc_gguf_meta_get_i64(const tc_gguf_file* f, const char* key, int64_t default_val) {
    if (!f || !key) return default_val;
    for (uint64_t i = 0; i < f->metadata_kv_count; ++i) {
        if (strcmp(f->kvs[i].key, key) == 0) {
            switch (f->kvs[i].type) {
                case GGUF_TYPE_UINT8:   return (int64_t)f->kvs[i].v.u8;
                case GGUF_TYPE_INT8:    return (int64_t)f->kvs[i].v.i8;
                case GGUF_TYPE_UINT16:  return (int64_t)f->kvs[i].v.u16;
                case GGUF_TYPE_INT16:   return (int64_t)f->kvs[i].v.i16;
                case GGUF_TYPE_UINT32:  return (int64_t)f->kvs[i].v.u32;
                case GGUF_TYPE_INT32:   return (int64_t)f->kvs[i].v.i32;
                case GGUF_TYPE_UINT64:  return (int64_t)f->kvs[i].v.u64;
                case GGUF_TYPE_INT64:   return f->kvs[i].v.i64;
                default: return default_val;
            }
        }
    }
    return default_val;
}

double tc_gguf_meta_get_f64(const tc_gguf_file* f, const char* key, double default_val) {
    if (!f || !key) return default_val;
    for (uint64_t i = 0; i < f->metadata_kv_count; ++i) {
        if (strcmp(f->kvs[i].key, key) == 0) {
            switch (f->kvs[i].type) {
                case GGUF_TYPE_UINT8:   return (double)f->kvs[i].v.u8;
                case GGUF_TYPE_INT8:    return (double)f->kvs[i].v.i8;
                case GGUF_TYPE_UINT16:  return (double)f->kvs[i].v.u16;
                case GGUF_TYPE_INT16:   return (double)f->kvs[i].v.i16;
                case GGUF_TYPE_UINT32:  return (double)f->kvs[i].v.u32;
                case GGUF_TYPE_INT32:   return (double)f->kvs[i].v.i32;
                case GGUF_TYPE_UINT64:  return (double)f->kvs[i].v.u64;
                case GGUF_TYPE_INT64:   return (double)f->kvs[i].v.i64;
                case GGUF_TYPE_FLOAT32: return (double)f->kvs[i].v.f32;
                case GGUF_TYPE_FLOAT64: return f->kvs[i].v.f64;
                default: return default_val;
            }
        }
    }
    return default_val;
}

static const gguf_kv* find_kv(const tc_gguf_file* f, const char* key) {
    if (!f || !key) return NULL;
    for (uint64_t i = 0; i < f->metadata_kv_count; ++i) {
        if (strcmp(f->kvs[i].key, key) == 0) return &f->kvs[i];
    }
    return NULL;
}

uint64_t tc_gguf_meta_array_count(const tc_gguf_file* f, const char* key) {
    const gguf_kv* kv = find_kv(f, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY) return 0;
    return kv->v.arr.n;
}

tc_status_t tc_gguf_meta_array_get_str(const tc_gguf_file* f,
                                       const char* key,
                                       uint64_t index,
                                       const char** out_ptr,
                                       size_t* out_len) {
    if (!out_ptr || !out_len) return TC_ERR_INVALID_ARG;
    *out_ptr = NULL;
    *out_len = 0;
    const gguf_kv* kv = find_kv(f, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY ||
        kv->v.arr.elem != GGUF_TYPE_STRING ||
        index >= kv->v.arr.n) return TC_ERR_INVALID_ARG;

    reader_t r = {
        (const uint8_t*)kv->v.arr.p,
        0,
        f->map_size - (size_t)((const uint8_t*)kv->v.arr.p - (const uint8_t*)f->map)
    };
    const char* p = NULL;
    uint64_t n = 0;
    for (uint64_t i = 0; i <= index; ++i) {
        if (rd_str(&r, &p, &n) != 0) return TC_ERR_INTERNAL;
    }
    if (n > SIZE_MAX) return TC_ERR_INTERNAL;
    *out_ptr = p;
    *out_len = (size_t)n;
    return TC_OK;
}

static int64_t scalar_at_i64(const void* p, gguf_value_type_t type, int64_t default_val) {
    switch (type) {
        case GGUF_TYPE_UINT8:  return (int64_t)*(const uint8_t*)p;
        case GGUF_TYPE_INT8:   return (int64_t)*(const int8_t*)p;
        case GGUF_TYPE_UINT16: { uint16_t v; memcpy(&v, p, 2); return (int64_t)v; }
        case GGUF_TYPE_INT16:  { int16_t  v; memcpy(&v, p, 2); return (int64_t)v; }
        case GGUF_TYPE_UINT32: { uint32_t v; memcpy(&v, p, 4); return (int64_t)v; }
        case GGUF_TYPE_INT32:  { int32_t  v; memcpy(&v, p, 4); return (int64_t)v; }
        case GGUF_TYPE_UINT64: { uint64_t v; memcpy(&v, p, 8); return (int64_t)v; }
        case GGUF_TYPE_INT64:  { int64_t  v; memcpy(&v, p, 8); return v; }
        case GGUF_TYPE_BOOL:   return (int64_t)*(const uint8_t*)p;
        default: return default_val;
    }
}

static double scalar_at_f64(const void* p, gguf_value_type_t type, double default_val) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return (double)*(const uint8_t*)p;
        case GGUF_TYPE_INT8:    return (double)*(const int8_t*)p;
        case GGUF_TYPE_UINT16:  { uint16_t v; memcpy(&v, p, 2); return (double)v; }
        case GGUF_TYPE_INT16:   { int16_t  v; memcpy(&v, p, 2); return (double)v; }
        case GGUF_TYPE_UINT32:  { uint32_t v; memcpy(&v, p, 4); return (double)v; }
        case GGUF_TYPE_INT32:   { int32_t  v; memcpy(&v, p, 4); return (double)v; }
        case GGUF_TYPE_UINT64:  { uint64_t v; memcpy(&v, p, 8); return (double)v; }
        case GGUF_TYPE_INT64:   { int64_t  v; memcpy(&v, p, 8); return (double)v; }
        case GGUF_TYPE_FLOAT32: { float    v; memcpy(&v, p, 4); return (double)v; }
        case GGUF_TYPE_FLOAT64: { double   v; memcpy(&v, p, 8); return v; }
        case GGUF_TYPE_BOOL:    return (double)*(const uint8_t*)p;
        default: return default_val;
    }
}

int64_t tc_gguf_meta_array_get_i64(const tc_gguf_file* f,
                                   const char* key,
                                   uint64_t index,
                                   int64_t default_val) {
    const gguf_kv* kv = find_kv(f, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY || index >= kv->v.arr.n) return default_val;
    const size_t elem_size = gguf_scalar_size(kv->v.arr.elem);
    if (elem_size == 0) return default_val;
    const void* p = (const uint8_t*)kv->v.arr.p + index * elem_size;
    return scalar_at_i64(p, kv->v.arr.elem, default_val);
}

double tc_gguf_meta_array_get_f64(const tc_gguf_file* f,
                                  const char* key,
                                  uint64_t index,
                                  double default_val) {
    const gguf_kv* kv = find_kv(f, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY || index >= kv->v.arr.n) return default_val;
    const size_t elem_size = gguf_scalar_size(kv->v.arr.elem);
    if (elem_size == 0) return default_val;
    const void* p = (const uint8_t*)kv->v.arr.p + index * elem_size;
    return scalar_at_f64(p, kv->v.arr.elem, default_val);
}

tc_status_t tc_gguf_get_llama_config(const tc_gguf_file* f,
                                     tc_gguf_llama_config* out_config) {
    if (!f || !out_config) return TC_ERR_INVALID_ARG;
    const char* arch = tc_gguf_meta_get_str(f, "general.architecture");
    if (!arch || strcmp(arch, "llama") != 0) return TC_ERR_INVALID_ARG;

    memset(out_config, 0, sizeof(*out_config));
    out_config->context_length =
        tc_gguf_meta_get_i64(f, "llama.context_length", 0);
    out_config->embedding_length =
        tc_gguf_meta_get_i64(f, "llama.embedding_length", 0);
    out_config->feed_forward_length =
        tc_gguf_meta_get_i64(f, "llama.feed_forward_length", 0);
    out_config->block_count =
        tc_gguf_meta_get_i64(f, "llama.block_count", 0);
    out_config->attention_head_count =
        tc_gguf_meta_get_i64(f, "llama.attention.head_count", 0);
    out_config->attention_head_count_kv =
        tc_gguf_meta_get_i64(f, "llama.attention.head_count_kv",
                             out_config->attention_head_count);
    out_config->rope_dimension_count =
        tc_gguf_meta_get_i64(f, "llama.rope.dimension_count", 0);
    out_config->vocab_size =
        (int64_t)tc_gguf_meta_array_count(f, "tokenizer.ggml.tokens");
    out_config->rms_norm_epsilon =
        tc_gguf_meta_get_f64(f, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    out_config->rope_freq_base =
        tc_gguf_meta_get_f64(f, "llama.rope.freq_base", 10000.0);
    out_config->rope_freq_scale =
        tc_gguf_meta_get_f64(f, "llama.rope.freq_scale", 1.0);
    return TC_OK;
}

tc_status_t tc_gguf_tensor_to_buffer(tc_context* ctx,
                                     const tc_gguf_file* f,
                                     const char* name,
                                     tc_buffer** out_buffer) {
    if (!ctx || !f || !name || !out_buffer) return TC_ERR_INVALID_ARG;
    *out_buffer = NULL;

    tc_gguf_tensor_info info;
    tc_status_t s = tc_gguf_get_tensor(f, name, &info);
    if (s != TC_OK) return s;
    if (!info.data || info.n_bytes == 0 || info.type == TC_GGUF_TYPE_UNSUPPORTED) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }

    return tc_gguf_tensor_info_to_buffer(ctx, &info, out_buffer);
}

static tc_status_t loaded_tensor_to_info(const loaded_tensor* t,
                                         tc_gguf_loaded_tensor_info* out_info) {
    if (!t || !out_info) return TC_ERR_INVALID_ARG;
    memset(out_info, 0, sizeof(*out_info));
    out_info->name = t->name;
    out_info->n_dims = t->n_dims;
    for (int32_t i = 0; i < t->n_dims && i < 4; ++i) {
        out_info->dims[i] = t->dims[i];
    }
    out_info->type = t->type;
    out_info->offset = t->offset;
    out_info->n_bytes = t->n_bytes;
    out_info->buffer = t->buffer;
    return TC_OK;
}

static tc_status_t tc_gguf_tensor_info_to_buffer(tc_context* ctx,
                                                 const tc_gguf_tensor_info* info,
                                                 tc_buffer** out_buffer) {
    if (!ctx || !info || !out_buffer) return TC_ERR_INVALID_ARG;
    *out_buffer = NULL;
    if (!info->data || info->n_bytes == 0 || info->type == TC_GGUF_TYPE_UNSUPPORTED) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }

    tc_buffer* buf = NULL;
    tc_status_t s = tc_buffer_alloc(ctx, info->n_bytes, &buf);
    if (s != TC_OK) return s;

    void* dst = NULL;
    s = tc_buffer_map(buf, &dst);
    if (s != TC_OK) {
        tc_buffer_free(ctx, buf);
        return s;
    }
    memcpy(dst, info->data, info->n_bytes);
    *out_buffer = buf;
    return TC_OK;
}

static tc_status_t gguf_type_to_quant(tc_gguf_type_t type, tc_quant_t* out) {
    if (!out) return TC_ERR_INVALID_ARG;
    switch (type) {
        case TC_GGUF_TYPE_Q4_0:
            *out = TC_QUANT_Q4_0;
            return TC_OK;
        case TC_GGUF_TYPE_Q8_0:
            *out = TC_QUANT_Q8_0;
            return TC_OK;
        default:
            return TC_ERR_UNSUPPORTED_DTYPE;
    }
}

static tc_status_t gguf_quantized_matrix_info_common(int32_t n_dims,
                                                     const uint64_t dims[4],
                                                     tc_gguf_type_t type,
                                                     size_t n_bytes,
                                                     tc_buffer* buffer,
                                                     tc_gguf_quantized_matrix_info* out_info) {
    if (!dims || !out_info) return TC_ERR_INVALID_ARG;
    memset(out_info, 0, sizeof(*out_info));

    tc_quant_t quant_type;
    tc_status_t s = gguf_type_to_quant(type, &quant_type);
    if (s != TC_OK) return s;

    if (n_dims != 2 || dims[0] == 0 || dims[1] == 0 ||
        dims[0] > (uint64_t)INT_MAX || dims[1] > (uint64_t)INT_MAX ||
        dims[0] % 32 != 0) {
        return TC_ERR_INVALID_SHAPE;
    }

    const int K = (int)dims[0];
    const int N = (int)dims[1];
    const size_t expected = tc_quantized_size(quant_type, N, K);
    if (expected == 0 || n_bytes != expected) {
        return TC_ERR_INVALID_SHAPE;
    }

    out_info->N = N;
    out_info->K = K;
    out_info->gguf_type = type;
    out_info->quant_type = quant_type;
    out_info->n_bytes = n_bytes;
    out_info->buffer = buffer;
    return TC_OK;
}

tc_status_t tc_gguf_tensor_quantized_matrix_info(
    const tc_gguf_tensor_info* tensor,
    tc_gguf_quantized_matrix_info* out_info) {
    if (!tensor || !out_info) return TC_ERR_INVALID_ARG;
    return gguf_quantized_matrix_info_common(tensor->n_dims, tensor->dims,
                                             tensor->type, tensor->n_bytes,
                                             NULL, out_info);
}

tc_status_t tc_gguf_loaded_tensor_quantized_matrix_info(
    const tc_gguf_loaded_tensor_info* tensor,
    tc_gguf_quantized_matrix_info* out_info) {
    if (!tensor || !out_info) return TC_ERR_INVALID_ARG;
    return gguf_quantized_matrix_info_common(tensor->n_dims, tensor->dims,
                                             tensor->type, tensor->n_bytes,
                                             tensor->buffer, out_info);
}

tc_status_t tc_gguf_load_supported_tensors(tc_context* ctx,
                                           const tc_gguf_file* f,
                                           tc_gguf_loaded_model** out_model) {
    if (!ctx || !f || !out_model) return TC_ERR_INVALID_ARG;
    *out_model = NULL;

    tc_gguf_loaded_model* model = (tc_gguf_loaded_model*)calloc(1, sizeof(*model));
    if (!model) return TC_ERR_ALLOC;

    if (f->tensor_count > SIZE_MAX / sizeof(loaded_tensor)) {
        free(model);
        return TC_ERR_ALLOC;
    }
    model->tensors = (loaded_tensor*)calloc(f->tensor_count, sizeof(loaded_tensor));
    if (f->tensor_count && !model->tensors) {
        free(model);
        return TC_ERR_ALLOC;
    }

    for (uint64_t i = 0; i < f->tensor_count; ++i) {
        const tc_gguf_tensor_info* src = &f->tensors[i];
        if (!src->data || src->n_bytes == 0 || src->type == TC_GGUF_TYPE_UNSUPPORTED) {
            model->skipped++;
            continue;
        }

        loaded_tensor* dst = &model->tensors[model->count];
        dst->name = gguf_strdup(src->name ? src->name : "");
        if (!dst->name) {
            tc_gguf_loaded_model_free(ctx, model);
            return TC_ERR_ALLOC;
        }
        dst->n_dims = src->n_dims;
        for (int32_t d = 0; d < src->n_dims && d < 4; ++d) {
            dst->dims[d] = src->dims[d];
        }
        dst->type = src->type;
        dst->offset = src->offset;
        dst->n_bytes = src->n_bytes;

        tc_status_t s = tc_gguf_tensor_info_to_buffer(ctx, src, &dst->buffer);
        if (s != TC_OK) {
            tc_gguf_loaded_model_free(ctx, model);
            return s;
        }
        model->count++;
    }

    *out_model = model;
    return TC_OK;
}

void tc_gguf_loaded_model_free(tc_context* ctx, tc_gguf_loaded_model* model) {
    if (!model) return;
    if (model->tensors) {
        for (uint64_t i = 0; i < model->count; ++i) {
            free(model->tensors[i].name);
            if (ctx && model->tensors[i].buffer) {
                tc_buffer_free(ctx, model->tensors[i].buffer);
            }
        }
        free(model->tensors);
    }
    free(model);
}

uint64_t tc_gguf_loaded_tensor_count(const tc_gguf_loaded_model* model) {
    return model ? model->count : 0;
}

uint64_t tc_gguf_loaded_skipped_tensor_count(const tc_gguf_loaded_model* model) {
    return model ? model->skipped : 0;
}

tc_status_t tc_gguf_loaded_tensor_at(const tc_gguf_loaded_model* model,
                                     uint64_t i,
                                     tc_gguf_loaded_tensor_info* out_info) {
    if (!model || !out_info || i >= model->count) return TC_ERR_INVALID_ARG;
    return loaded_tensor_to_info(&model->tensors[i], out_info);
}

tc_status_t tc_gguf_loaded_get_tensor(const tc_gguf_loaded_model* model,
                                      const char* name,
                                      tc_gguf_loaded_tensor_info* out_info) {
    if (!model || !name || !out_info) return TC_ERR_INVALID_ARG;
    for (uint64_t i = 0; i < model->count; ++i) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return loaded_tensor_to_info(&model->tensors[i], out_info);
        }
    }
    return TC_ERR_INVALID_ARG;
}
