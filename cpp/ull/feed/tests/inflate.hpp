// Minimal gzip reader for the tests (RFC 1952 header, RFC 1951 inflate).
//
// The live recordings in tools/recorder/recordings are gzip files and the
// feed library has no dependencies, so the test carries its own inflater.
// Canonical Huffman decoding by code-length counts. Not for production: it
// trusts its input and throws std::runtime_error on anything it cannot read.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qp::ull::feed::gz {

class Inflater {
public:
    explicit Inflater(std::string_view b, std::size_t at) : b_(b), i_(at) {}

    std::string run() {
        std::string out;
        bool last = false;
        while (!last) {
            last = take(1) == 1;
            switch (take(2)) {
                case 0:
                    stored(out);
                    break;
                case 1: {
                    std::vector<std::uint8_t> l(288, 8);
                    for (int k = 144; k < 256; ++k) l[std::size_t(k)] = 9;
                    for (int k = 256; k < 280; ++k) l[std::size_t(k)] = 7;
                    block(make(l), make(std::vector<std::uint8_t>(30, 5)), out);
                    break;
                }
                case 2:
                    dynamic(out);
                    break;
                default:
                    throw std::runtime_error("bad deflate block");
            }
        }
        return out;
    }

private:
    struct Huff {
        std::array<int, 16> count{};
        std::vector<int> sym;
    };

    std::string_view b_;
    std::size_t i_;
    std::uint32_t buf_ = 0;
    unsigned n_ = 0;

    std::uint32_t byte_at(std::size_t k) const {
        if (k >= b_.size()) throw std::runtime_error("short gzip stream");
        return static_cast<unsigned char>(b_[k]);
    }
    std::uint32_t take(unsigned k) {
        while (n_ < k) {
            buf_ |= byte_at(i_++) << n_;
            n_ += 8;
        }
        const std::uint32_t v = buf_ & ((1u << k) - 1u);
        buf_ >>= k;
        n_ -= k;
        return v;
    }
    static Huff make(const std::vector<std::uint8_t>& lens) {
        Huff h;
        for (auto l : lens) ++h.count[l];
        h.count[0] = 0;
        for (std::uint8_t len = 1; len < 16; ++len)
            for (std::size_t s = 0; s < lens.size(); ++s)
                if (lens[s] == len) h.sym.push_back(int(s));
        return h;
    }
    int sym(const Huff& h) {
        int code = 0, first = 0, index = 0;
        for (std::size_t len = 1; len < 16; ++len) {
            code |= int(take(1));
            const int c = h.count[len];
            if (code - c < first) return h.sym[std::size_t(index + code - first)];
            index += c;
            first = (first + c) << 1;
            code <<= 1;
        }
        throw std::runtime_error("bad Huffman code");
    }
    void stored(std::string& out) {
        buf_ = 0;
        n_ = 0;
        const std::size_t n = byte_at(i_) | (byte_at(i_ + 1) << 8);
        i_ += 4;
        out.append(b_.substr(i_, n));
        i_ += n;
    }
    void dynamic(std::string& out) {
        static constexpr std::array<std::size_t, 19> order{16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                                           11, 4,  12, 3, 13, 2, 14, 1, 15};
        const std::size_t nlen = take(5) + 257, ndist = take(5) + 1, ncode = take(4) + 4;
        std::vector<std::uint8_t> cl(19, 0);
        for (std::size_t k = 0; k < ncode; ++k) cl[order[k]] = std::uint8_t(take(3));
        const Huff h = make(cl);
        std::vector<std::uint8_t> l;
        while (l.size() < nlen + ndist) {
            const int s = sym(h);
            if (s < 16) {
                l.push_back(std::uint8_t(s));
            } else if (s == 16) {
                const std::uint8_t p = l.at(l.size() - 1);
                l.insert(l.end(), 3 + take(2), p);
            } else if (s == 17) {
                l.insert(l.end(), 3 + take(3), std::uint8_t(0));
            } else {
                l.insert(l.end(), 11 + take(7), std::uint8_t(0));
            }
        }
        const auto split = l.begin() + std::ptrdiff_t(nlen);
        block(make({l.begin(), split}), make({split, l.end()}), out);
    }
    void block(const Huff& lit, const Huff& dist, std::string& out) {
        static constexpr std::array<std::size_t, 29> len{
            3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
            31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
        static constexpr std::array<unsigned, 29> lext{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                                       2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
        static constexpr std::array<std::size_t, 30> dst{
            1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
            193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
        static constexpr std::array<unsigned, 30> dext{0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                                       4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                                       9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
        for (;;) {
            const int s = sym(lit);
            if (s < 256) {
                out.push_back(char(s));
            } else if (s == 256) {
                return;
            } else {
                const auto k = std::size_t(s - 257);
                const std::size_t n = len.at(k) + take(lext.at(k));
                const auto d = std::size_t(sym(dist));
                const std::size_t back = dst.at(d) + take(dext.at(d));
                for (std::size_t j = 0; j < n; ++j) out.push_back(out.at(out.size() - back));
            }
        }
    }
};

inline std::string gunzip(std::string_view b) {
    if (b.size() < 18 || b[0] != '\x1f' || b[1] != '\x8b' || b[2] != 8)
        throw std::runtime_error("not a gzip file");
    const auto flags = static_cast<unsigned char>(b[3]);
    std::size_t i = 10;
    if (flags & 4)
        i += 2 + static_cast<unsigned char>(b[i]) +
             (std::size_t(static_cast<unsigned char>(b[i + 1])) << 8);
    for (unsigned bit : {8u, 16u})
        if (flags & bit) {
            while (b.at(i) != 0) ++i;
            ++i;
        }
    if (flags & 2) i += 2;
    std::string out = Inflater(b, i).run();
    std::uint32_t isize = 0;
    for (std::size_t k = 0; k < 4; ++k)
        isize |= std::uint32_t(static_cast<unsigned char>(b[b.size() - 4 + k])) << (8 * k);
    if (isize != static_cast<std::uint32_t>(out.size()))
        throw std::runtime_error("gzip length mismatch");
    return out;
}

}  // namespace qp::ull::feed::gz
