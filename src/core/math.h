// Probability helpers shared by every module.
#pragma once

#include <cstdint>

namespace cmix {

constexpr int kProbOne = 65536;
constexpr int kProbMin = 1;
constexpr int kProbMax = 65535;

inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

float stretch(int p);
float squash(float x);
int to_p16(float p);
double bit_cost(int p, int bit);

}  // namespace cmix
