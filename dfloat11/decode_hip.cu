// DFloat11 decode kernel — optimized for AMD RDNA/CDNA GPUs.
// Based on decode.cu by Tianyi Zhang (Apache 2.0).
//
// Changes from upstream CUDA version:
//   - LUT preloaded into shared memory (LDS) for guaranteed low-latency access
//   - __ldg() removed (AMD caches global reads by default)
//   - volatile removed from shared memory (LDS writes are immediately visible
//     to all threads in a workgroup without volatile on AMD)

#define THREAD_ID           threadIdx.x
#define BLOCK_ID            blockIdx.x
#define N_THREADS           blockDim.x
#define BYTES_PER_THREAD    8
#define MAX_LUT_BYTES       2048

typedef unsigned char       uint8_t;
typedef unsigned short      uint16_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

extern "C"
__global__ void decode(
    const uint8_t * __restrict__    luts,
    const uint8_t * __restrict__    codes,
    const uint8_t * __restrict__    sign_mantissa,
    const uint32_t * __restrict__   position_offsets,
    const uint8_t * __restrict__    gaps,
    uint16_t * __restrict__         outputs,
    const int n_luts, const int n_bytes, const int n_elements
) {
    uint8_t register_buffer[12];
    extern __shared__ uint8_t shared_mem[];

    // Shared memory layout:
    //   [0 .. lut_bytes)                          — LUT (preloaded from global)
    //   [lut_bytes .. lut_bytes + N_THREADS*4+4)  — accumulators
    //   [lut_bytes + N_THREADS*4+4 .. end)        — write_buffer
    const int lut_bytes = n_luts * 256;
    uint8_t*  shared_luts    = shared_mem;
    uint32_t* accumulators   = (uint32_t*)(shared_mem + lut_bytes);
    uint16_t* write_buffer   = (uint16_t*)(shared_mem + lut_bytes + N_THREADS * 4 + 4);

    // Cooperatively load LUT into shared memory
    for (int j = THREAD_ID; j < lut_bytes; j += N_THREADS) {
        shared_luts[j] = luts[j];
    }
    __syncthreads();

    const int global_thread_id = BLOCK_ID * N_THREADS + THREAD_ID;

    if (global_thread_id * BYTES_PER_THREAD < n_bytes) {
        register_buffer[0] = codes[global_thread_id * BYTES_PER_THREAD];
    }
    if (global_thread_id * BYTES_PER_THREAD + 1 < n_bytes) {
        register_buffer[1] = codes[global_thread_id * BYTES_PER_THREAD + 1];
    }
    if (global_thread_id * BYTES_PER_THREAD + 2 < n_bytes) {
        register_buffer[2] = codes[global_thread_id * BYTES_PER_THREAD + 2];
    }
    if (global_thread_id * BYTES_PER_THREAD + 3 < n_bytes) {
        register_buffer[3] = codes[global_thread_id * BYTES_PER_THREAD + 3];
    }
    if (global_thread_id * BYTES_PER_THREAD + 4 < n_bytes) {
        register_buffer[4] = codes[global_thread_id * BYTES_PER_THREAD + 4];
    }
    if (global_thread_id * BYTES_PER_THREAD + 5 < n_bytes) {
        register_buffer[5] = codes[global_thread_id * BYTES_PER_THREAD + 5];
    }
    if (global_thread_id * BYTES_PER_THREAD + 6 < n_bytes) {
        register_buffer[6] = codes[global_thread_id * BYTES_PER_THREAD + 6];
    }
    if (global_thread_id * BYTES_PER_THREAD + 7 < n_bytes) {
        register_buffer[7] = codes[global_thread_id * BYTES_PER_THREAD + 7];
    }
    if (global_thread_id * BYTES_PER_THREAD + 8 < n_bytes) {
        register_buffer[8] = codes[global_thread_id * BYTES_PER_THREAD + 8];
    }
    if (global_thread_id * BYTES_PER_THREAD + 9 < n_bytes) {
        register_buffer[9] = codes[global_thread_id * BYTES_PER_THREAD + 9];
    }
    if (global_thread_id * BYTES_PER_THREAD + 10 < n_bytes) {
        register_buffer[10] = codes[global_thread_id * BYTES_PER_THREAD + 10];
    }
    if (global_thread_id * BYTES_PER_THREAD + 11 < n_bytes) {
        register_buffer[11] = codes[global_thread_id * BYTES_PER_THREAD + 11];
    }
    __syncthreads();

    alignas(8) uint8_t buffer[12];
    uint64_t &long_buffer   = *reinterpret_cast<uint64_t *>(buffer);
    uint32_t &int_buffer    = *reinterpret_cast<uint32_t *>(buffer + 8);
    uint16_t &short_buffer  = *reinterpret_cast<uint16_t *>(buffer + 8);

    buffer[8] = gaps[global_thread_id * 5 / 8 + 1];
    buffer[9] = gaps[global_thread_id * 5 / 8];
    const uint8_t gap = (short_buffer >> (11 - (global_thread_id * 5 % 8))) & 0x1f;

    uint32_t thread_counter = 0;

    buffer[0] = register_buffer[7];
    buffer[1] = register_buffer[6];
    buffer[2] = register_buffer[5];
    buffer[3] = register_buffer[4];
    buffer[4] = register_buffer[3];
    buffer[5] = register_buffer[2];
    buffer[6] = register_buffer[1];
    buffer[7] = register_buffer[0];

    long_buffer <<= gap;
    uint8_t free_bits = gap;
    uint8_t decoded;

    while (free_bits < 32) {
        decoded = shared_luts[long_buffer >> 56];
        if (decoded >= 240) {
            decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)];
            if (decoded >= 240) {
                decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 40) & 0xff)];
                if (decoded >= 240) {
                    decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 32) & 0xff)];
                }
            }
        }
        thread_counter += 1;
        decoded = shared_luts[256 * (n_luts - 1) + decoded];
        long_buffer <<= decoded;
        free_bits += decoded;
    }

    buffer[8]   = register_buffer[11];
    buffer[9]   = register_buffer[10];
    buffer[10]  = register_buffer[9];
    buffer[11]  = register_buffer[8];

    long_buffer |= static_cast<uint64_t>(int_buffer) << (free_bits - 32);
    free_bits -= 32;

    while (4 + free_bits / 8 < BYTES_PER_THREAD) {
        decoded = shared_luts[long_buffer >> 56];
        if (decoded >= 240) {
            decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)];
            if (decoded >= 240) {
                decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 40) & 0xff)];
                if (decoded >= 240) {
                    decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 32) & 0xff)];
                }
            }
        }
        thread_counter += 1;
        decoded = shared_luts[256 * (n_luts - 1) + decoded];
        long_buffer <<= decoded;
        free_bits += decoded;
    }

    if (THREAD_ID == 0) {
        accumulators[0] = position_offsets[BLOCK_ID] + thread_counter;
    } else {
        accumulators[THREAD_ID] = thread_counter;
    }
    __syncthreads();

    int i;
    for (i = 2; i <= N_THREADS; i <<= 1) {
        if (((THREAD_ID + 1) & (i - 1)) == 0) {
            accumulators[THREAD_ID] += accumulators[THREAD_ID - (i >> 1)];
        }
        __syncthreads();
    }

    if (THREAD_ID == 0) {
        accumulators[N_THREADS - 1] = 0;
    }
    __syncthreads();

    for (i = N_THREADS; i >= 2; i >>= 1) {
        if (((THREAD_ID + 1) & (i - 1)) == 0) {
            accumulators[THREAD_ID] += accumulators[THREAD_ID - (i >> 1)];
            accumulators[THREAD_ID - (i >> 1)] = accumulators[THREAD_ID] - accumulators[THREAD_ID - (i >> 1)];
        }
        __syncthreads();
    }

    if (THREAD_ID == 0) {
        accumulators[0] = position_offsets[BLOCK_ID];
        accumulators[N_THREADS] = position_offsets[BLOCK_ID+1];
    }
    __syncthreads();

    uint32_t output_idx = accumulators[THREAD_ID], write_offset = accumulators[0];
    const uint32_t end_output_idx = min(output_idx + thread_counter, n_elements);

    buffer[0] = register_buffer[7];
    buffer[1] = register_buffer[6];
    buffer[2] = register_buffer[5];
    buffer[3] = register_buffer[4];
    buffer[4] = register_buffer[3];
    buffer[5] = register_buffer[2];
    buffer[6] = register_buffer[1];
    buffer[7] = register_buffer[0];

    long_buffer <<= gap;
    free_bits = gap;

    while (free_bits < 32 && output_idx < end_output_idx) {
        decoded = shared_luts[long_buffer >> 56];
        if (decoded >= 240) {
            decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)];
            if (decoded >= 240) {
                decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 40) & 0xff)];
                if (decoded >= 240) {
                    decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 32) & 0xff)];
                }
            }
        }

        buffer[8] = sign_mantissa[output_idx];
        buffer[9] = (buffer[8] & 128) | (decoded >> 1);
        buffer[8] = (decoded << 7) | (buffer[8] & 127);
        write_buffer[output_idx - write_offset] = short_buffer;

        output_idx += 1;
        decoded = shared_luts[256 * (n_luts - 1) + decoded];
        long_buffer <<= decoded;
        free_bits += decoded;
    }

    buffer[8]   = register_buffer[11];
    buffer[9]   = register_buffer[10];
    buffer[10]  = register_buffer[9];
    buffer[11]  = register_buffer[8];

    long_buffer |= static_cast<uint64_t>(int_buffer) << (free_bits - 32);
    free_bits -= 32;

    while (output_idx < end_output_idx) {
        decoded = shared_luts[long_buffer >> 56];
        if (decoded >= 240) {
            decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)];
            if (decoded >= 240) {
                decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 40) & 0xff)];
                if (decoded >= 240) {
                    decoded = shared_luts[256 * (256 - decoded) + ((long_buffer >> 32) & 0xff)];
                }
            }
        }

        buffer[8] = sign_mantissa[output_idx];
        buffer[9] = (buffer[8] & 128) | (decoded >> 1);
        buffer[8] = (decoded << 7) | (buffer[8] & 127);
        write_buffer[output_idx - write_offset] = short_buffer;

        output_idx += 1;
        decoded = shared_luts[256 * (n_luts - 1) + decoded];
        long_buffer <<= decoded;
        free_bits += decoded;
    }
    __syncthreads();

    for (i = THREAD_ID; i < min(accumulators[N_THREADS] - write_offset, n_elements - write_offset); i += N_THREADS) {
        outputs[i + write_offset] = write_buffer[i];
    }
}
