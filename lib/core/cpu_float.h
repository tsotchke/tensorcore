#ifndef TC_CPU_FLOAT_H
#define TC_CPU_FLOAT_H

#include <cmath>
#include <cstdint>
#include <cstring>

static inline float tc_cpu_f32_from_bits(uint32_t bits) {
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint32_t tc_cpu_f32_to_bits(float value) {
    uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

static inline float tc_cpu_f16_to_f32(uint16_t bits) {
    const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t mant = bits & 0x03ffu;

    if (exp == 0) {
        if (mant == 0) return tc_cpu_f32_from_bits(sign);
        int e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03ffu;
        return tc_cpu_f32_from_bits(sign |
                                    (uint32_t)(e + 127) << 23 |
                                    (mant << 13));
    }
    if (exp == 0x1fu) {
        return tc_cpu_f32_from_bits(sign | 0x7f800000u | (mant << 13));
    }
    return tc_cpu_f32_from_bits(sign |
                                ((exp + (127u - 15u)) << 23) |
                                (mant << 13));
}

static inline uint16_t tc_cpu_f32_to_f16(float value) {
    const uint32_t bits = tc_cpu_f32_to_bits(value);
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;

    if (exp == 0xffu) {
        if (mant == 0) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | 0x7e00u);
    }

    int half_exp = (int)exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        const uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) +
                                 ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }

    uint32_t rounded = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0;
        ++half_exp;
        if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

static inline float tc_cpu_bf16_to_f32(uint16_t bits) {
    return tc_cpu_f32_from_bits((uint32_t)bits << 16);
}

static inline uint16_t tc_cpu_f32_to_bf16(float value) {
    uint32_t bits = tc_cpu_f32_to_bits(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

#endif /* TC_CPU_FLOAT_H */
