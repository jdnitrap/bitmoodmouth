// Hash mixing for context keys.
#pragma once

#include <cstdint>

namespace cmix {

inline uint32_t hash_mix(uint32_t h, uint32_t v) {
  h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
  return h * 0x85ebca6bu;
}

inline uint64_t hash_fin(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

inline uint64_t hash_add(uint64_t h, uint64_t v) { return hash_fin(h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6))); }

}  // namespace cmix
