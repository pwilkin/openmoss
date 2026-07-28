// SPDX-License-Identifier: Apache-2.0
//
// Minimal RIFF/WAVE reader & writer.
//
// Multi-channel buffers are interleaved; see openmoss/wav.h.
//
// We deliberately avoid pulling in libsndfile / dr_wav. WAV is simple enough to
// hand-parse and keeping the dependency surface small means the released
// binary is one self-contained ELF.

#include "openmoss/wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace openmoss {

namespace {

struct ChunkHeader { char id[4]; uint32_t size; };

uint16_t read_u16_le(const uint8_t * p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t read_u32_le(const uint8_t * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8)
         | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
void write_u16_le(uint8_t * p, uint16_t v) { p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; }
void write_u32_le(uint8_t * p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = (v >> 24) & 0xff;
}

// Naive linear-interpolation resampler. Good enough for prompt audio; we are
// not in a quality-sensitive path here (the codec sees 24 kHz internally).
std::vector<float> resample_linear(const std::vector<float> & in, int32_t in_sr, int32_t out_sr) {
    if (in_sr == out_sr || in.empty()) return in;
    const double ratio = double(in_sr) / double(out_sr);
    const int64_t n_out = int64_t(std::llround(in.size() / ratio));
    std::vector<float> out(n_out);
    for (int64_t i = 0; i < n_out; ++i) {
        const double s   = i * ratio;
        const int64_t s0 = int64_t(std::floor(s));
        const int64_t s1 = std::min<int64_t>(s0 + 1, int64_t(in.size()) - 1);
        const float   f  = float(s - s0);
        out[i] = in[s0] * (1.f - f) + in[s1] * f;
    }
    return out;
}

} // namespace

std::vector<float> decode_wav(const uint8_t * data,
                              size_t          n_bytes,
                              int32_t         target_sr,
                              int32_t         want_channels) {
    if (want_channels < 1) throw std::runtime_error("decode_wav: want_channels must be >= 1");
    if (n_bytes < 44) throw std::runtime_error("decode_wav: buffer too small");
    if (std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0)
        throw std::runtime_error("decode_wav: not a RIFF/WAVE buffer");

    uint16_t fmt_tag = 0, n_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> pcm_data;
    bool fmt_seen = false, data_seen = false;

    size_t p = 12;
    while (!data_seen) {
        if (p + 8 > n_bytes) throw std::runtime_error("decode_wav: truncated stream");
        const uint8_t * id = data + p;
        const uint32_t sz  = read_u32_le(data + p + 4);
        p += 8;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (sz < 16 || p + sz > n_bytes)
                throw std::runtime_error("decode_wav: truncated fmt chunk");
            fmt_tag         = read_u16_le(data + p + 0);
            n_channels      = read_u16_le(data + p + 2);
            sample_rate     = read_u32_le(data + p + 4);
            bits_per_sample = read_u16_le(data + p + 14);
            fmt_seen = true;
            p += sz;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!fmt_seen) throw std::runtime_error("decode_wav: data before fmt");
            if (p + sz > n_bytes) throw std::runtime_error("decode_wav: truncated data");
            pcm_data.assign(data + p, data + p + sz);
            data_seen = true;
            p += sz;
        } else {
            // Skip unknown chunks; respect WAV's odd-byte alignment.
            const size_t skip = sz + (sz & 1u);
            if (p + skip > n_bytes) throw std::runtime_error("decode_wav: truncated chunk");
            p += skip;
        }
    }
    if (n_channels == 0) throw std::runtime_error("decode_wav: zero channels");

    const int64_t bytes_per_sample = bits_per_sample / 8;
    const int64_t n_frames = int64_t(pcm_data.size()) / (bytes_per_sample * n_channels);

    // Decode every source channel first, then adapt the channel count.
    // (Named counts, not `size_t(...)` temporaries — the latter would be parsed
    // as a function declaration.)
    const size_t n_src_ch  = size_t(n_channels);
    const size_t n_want_ch = size_t(want_channels);
    std::vector<std::vector<float>> src(n_src_ch);
    for (auto & ch : src) ch.resize(size_t(n_frames));

    for (int64_t i = 0; i < n_frames; ++i) {
        const uint8_t * frame = pcm_data.data() + i * bytes_per_sample * n_channels;
        for (uint16_t c = 0; c < n_channels; ++c) {
            const uint8_t * sp = frame + c * bytes_per_sample;
            float v = 0.f;
            if (fmt_tag == 1 /* PCM */) {
                if (bits_per_sample == 16) {
                    const int16_t s = int16_t(read_u16_le(sp));
                    v = s / 32768.0f;
                } else if (bits_per_sample == 24) {
                    int32_t s = (int32_t(sp[0])) | (int32_t(sp[1]) << 8) | (int32_t(sp[2]) << 16);
                    if (s & 0x800000) s |= int32_t(0xff000000);
                    v = s / 8388608.0f;
                } else if (bits_per_sample == 32) {
                    const int32_t s = int32_t(read_u32_le(sp));
                    v = s / 2147483648.0f;
                } else {
                    throw std::runtime_error("decode_wav: unsupported PCM bit depth");
                }
            } else if (fmt_tag == 3 /* IEEE float */) {
                if (bits_per_sample == 32) {
                    uint32_t bits = read_u32_le(sp);
                    std::memcpy(&v, &bits, 4);
                } else {
                    throw std::runtime_error("decode_wav: unsupported float bit depth");
                }
            } else if (fmt_tag == 0xFFFE /* WAVE_FORMAT_EXTENSIBLE */) {
                if (bits_per_sample == 16) {
                    const int16_t s = int16_t(read_u16_le(sp));
                    v = s / 32768.0f;
                } else {
                    throw std::runtime_error("decode_wav: unsupported EXTENSIBLE bit depth");
                }
            } else {
                throw std::runtime_error("decode_wav: unsupported wFormatTag");
            }
            src[c][size_t(i)] = v;
        }
    }

    // Adapt channel count: average down to mono, take the leading channels when
    // the file has more than asked for, duplicate channel 0 when it has fewer.
    std::vector<std::vector<float>> out_ch(n_want_ch);
    if (want_channels == 1 && n_channels > 1) {
        out_ch[0].resize(size_t(n_frames));
        for (int64_t i = 0; i < n_frames; ++i) {
            double acc = 0.0;
            for (uint16_t c = 0; c < n_channels; ++c) acc += src[c][size_t(i)];
            out_ch[0][size_t(i)] = float(acc / n_channels);
        }
    } else {
        for (int32_t c = 0; c < want_channels; ++c) {
            out_ch[size_t(c)] = (c < int32_t(n_channels)) ? src[size_t(c)] : src[0];
        }
    }

    if (sample_rate != uint32_t(target_sr)) {
        for (auto & ch : out_ch) ch = resample_linear(ch, int32_t(sample_rate), target_sr);
    }

    if (want_channels == 1) return out_ch[0];

    // Interleave.
    const size_t n_out = out_ch[0].size();
    std::vector<float> out(n_out * size_t(want_channels));
    for (size_t i = 0; i < n_out; ++i) {
        for (int32_t c = 0; c < want_channels; ++c) {
            out[i * size_t(want_channels) + size_t(c)] = out_ch[size_t(c)][i];
        }
    }
    return out;
}

std::vector<float> decode_wav_mono(const uint8_t * data, size_t n_bytes,
                                   int32_t target_sr) {
    return decode_wav(data, n_bytes, target_sr, 1);
}

std::vector<float> read_wav(const std::string & path, int32_t target_sr,
                            int32_t want_channels) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("read_wav: cannot open " + path);
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); throw std::runtime_error("read_wav: seek failed"); }
    long sz = std::ftell(f);
    std::rewind(f);
    if (sz < 0) { std::fclose(f); throw std::runtime_error("read_wav: ftell failed"); }
    std::vector<uint8_t> buf;
    buf.resize(size_t(sz));
    if (std::fread(buf.data(), 1, size_t(sz), f) != size_t(sz)) {
        std::fclose(f);
        throw std::runtime_error("read_wav: short read on " + path);
    }
    std::fclose(f);
    try {
        return decode_wav(buf.data(), buf.size(), target_sr, want_channels);
    } catch (const std::exception & e) {
        throw std::runtime_error("read_wav(" + path + "): " + e.what());
    }
}

std::vector<float> read_wav_mono(const std::string & path, int32_t target_sr) {
    return read_wav(path, target_sr, 1);
}

std::vector<uint8_t> encode_pcm_s16le(const float * pcm, int64_t n_samples) {
    if (n_samples < 0) n_samples = 0;
    std::vector<uint8_t> out(size_t(n_samples) * 2);
    for (int64_t i = 0; i < n_samples; ++i) {
        float v = pcm[i];
        v = std::max(-1.0f, std::min(1.0f, v));
        const int16_t s = int16_t(std::lround(v * 32767.0f));
        write_u16_le(out.data() + i * 2, uint16_t(s));
    }
    return out;
}

std::vector<uint8_t> encode_wav(const float * pcm,
                                int64_t       n_samples,
                                int32_t       sample_rate,
                                int32_t       n_channels) {
    if (n_samples < 0) n_samples = 0;
    if (n_channels < 1) throw std::runtime_error("encode_wav: n_channels must be >= 1");

    const uint16_t block_align  = uint16_t(2 * n_channels);   // 16-bit samples
    const uint32_t byte_rate    = uint32_t(sample_rate) * block_align;
    const uint32_t data_bytes   = uint32_t(n_samples) * 2;
    const uint32_t riff_payload = 36 + data_bytes;

    std::vector<uint8_t> out(size_t(44) + size_t(data_bytes));
    uint8_t * hdr = out.data();
    std::memcpy(hdr + 0, "RIFF", 4);
    write_u32_le(hdr + 4, riff_payload);
    std::memcpy(hdr + 8, "WAVE", 4);
    std::memcpy(hdr + 12, "fmt ", 4);
    write_u32_le(hdr + 16, 16);
    write_u16_le(hdr + 20, 1);   // PCM
    write_u16_le(hdr + 22, uint16_t(n_channels));
    write_u32_le(hdr + 24, uint32_t(sample_rate));
    write_u32_le(hdr + 28, byte_rate);
    write_u16_le(hdr + 32, block_align);
    write_u16_le(hdr + 34, 16);  // bits per sample
    std::memcpy(hdr + 36, "data", 4);
    write_u32_le(hdr + 40, data_bytes);

    const auto payload = encode_pcm_s16le(pcm, n_samples);
    std::memcpy(out.data() + 44, payload.data(), payload.size());
    return out;
}

std::vector<uint8_t> encode_wav_mono(const float * pcm, int64_t n_samples,
                                     int32_t sample_rate) {
    return encode_wav(pcm, n_samples, sample_rate, 1);
}

void write_wav(const std::string & path,
               const float *       pcm,
               int64_t             n_samples,
               int32_t             sample_rate,
               int32_t             n_channels) {
    auto bytes = encode_wav(pcm, n_samples, sample_rate, n_channels);
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("write_wav: cannot open " + path);
    if (std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        std::fclose(f);
        throw std::runtime_error("write_wav: write failed");
    }
    std::fclose(f);
}

void write_wav_mono(const std::string & path, const float * pcm,
                    int64_t n_samples, int32_t sample_rate) {
    write_wav(path, pcm, n_samples, sample_rate, 1);
}

} // namespace openmoss
