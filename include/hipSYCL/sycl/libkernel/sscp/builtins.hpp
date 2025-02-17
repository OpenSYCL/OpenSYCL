/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause


#ifndef ACPP_LIBKERNEL_SSCP_BUILTINS_HPP
#define ACPP_LIBKERNEL_SSCP_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "builtins/math.hpp"
#include "builtins/native.hpp"
#include "builtins/integer.hpp"
#include "builtins/relational.hpp"
#include "builtins/print.hpp"

#include <cstdlib>
#include <cmath>
#include <type_traits>

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP

namespace hipsycl {
namespace sycl {
namespace detail::sscp_builtins {

// ********************** math builtins *********************

#define HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(name)                        \
  HIPSYCL_BUILTIN float __acpp_##name(float x) {                            \
    return __acpp_sscp_##name##_f32(x);                                     \
  }                                                                            \
  HIPSYCL_BUILTIN double __acpp_##name(double x) {                          \
    return __acpp_sscp_##name##_f64(x);                                     \
  }

#define HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(name)                       \
  HIPSYCL_BUILTIN float __acpp_##name(float x, float y) {                   \
    return __acpp_sscp_##name##_f32(x, y);                                  \
  }                                                                            \
  HIPSYCL_BUILTIN double __acpp_##name(double x, double y) {                \
    return __acpp_sscp_##name##_f64(x, y);                                  \
  }

#define HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN3(name)                       \
  HIPSYCL_BUILTIN float __acpp_##name(float x, float y, float z) {          \
    return __acpp_sscp_##name##_f32(x, y, z);                               \
  }                                                                            \
  HIPSYCL_BUILTIN double __acpp_##name(double x, double y, double z) {      \
    return __acpp_sscp_##name##_f64(x, y, z);                               \
  }

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(acos)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(acosh)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(acospi)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(asin)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(asinh)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(asinpi)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(atan)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(atan2)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(atanh)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(atanpi)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(atan2pi)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(cbrt)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(ceil)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(copysign)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(cos)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(cosh)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(cospi)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(erf)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(erfc)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(exp)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(exp2)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(exp10)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(pow)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(expm1)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(fabs)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(fdim)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(floor)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN3(fma)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(fmax)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(fmin)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(fmod)

template<class T>
HIPSYCL_BUILTIN float __acpp_fract(float x, T* ptr) {
  float val;
  float res = __acpp_sscp_fract_f32(x, &val);
  *ptr = static_cast<T>(val);
  return res;
}

template<class T>
HIPSYCL_BUILTIN double __acpp_fract(double x, T* ptr) {
  double val;
  double res = __acpp_sscp_fract_f64(x, &val);
  *ptr = static_cast<T>(val);
  return res;
}

template<class IntT>
HIPSYCL_BUILTIN float __acpp_frexp(float x, IntT* ptr) {
  __acpp_int32 val;
  float res = __acpp_sscp_frexp_f32(x, &val);
  *ptr = static_cast<IntT>(val);
  return res;
}

template<class IntT>
HIPSYCL_BUILTIN double __acpp_frexp(double x, IntT* ptr) {
  __acpp_int32 val;
  double res = __acpp_sscp_frexp_f64(x, &val);
  *ptr = static_cast<IntT>(val);
  return res;
}

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(hypot)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(ilogb)

template<class IntType>
HIPSYCL_BUILTIN float __acpp_ldexp(float x, IntType k) noexcept {
  return __acpp_sscp_ldexp_f32(x, static_cast<__acpp_int32>(k));
}

template<class IntType>
HIPSYCL_BUILTIN double __acpp_ldexp(double x, IntType k) noexcept {
  return __acpp_sscp_ldexp_f64(x, static_cast<__acpp_int32>(k));
}

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(lgamma)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(tgamma)

template<class IntT>
HIPSYCL_BUILTIN float __acpp_lgamma_r(float x, IntT* ptr) {
  __acpp_int32 val;
  float res = __acpp_sscp_lgamma_r_f32(x, &val);
  *ptr = static_cast<IntT>(val);
  return res;
}

template<class IntT>
HIPSYCL_BUILTIN double __acpp_lgamma_r(double x, IntT* ptr) {
  __acpp_int32 val;
  double res = __acpp_sscp_lgamma_r_f64(x, &val);
  *ptr = static_cast<IntT>(val);
  return res;
}

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(log)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(log2)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(log10)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(log1p)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(logb)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN3(mad)


template<class T>
HIPSYCL_BUILTIN T __acpp_maxmag(T x, T y) noexcept {
  auto abs_x = (x < 0) ? -x : x;
  auto abs_y = (y < 0) ? -y : y;
  if(abs_x == abs_y) return sscp_builtins::__acpp_fmax(x,y);
  return (abs_x > abs_y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_minmag(T x, T y) noexcept {
  auto abs_x = (x < 0) ? -x : x;
  auto abs_y = (y < 0) ? -y : y;
  if(abs_x == abs_y) return sscp_builtins::__acpp_fmin(x,y);
  return (abs_x < abs_y) ? x : y;
}


template<class FloatT>
HIPSYCL_BUILTIN float __acpp_modf(float x, FloatT* y) noexcept {
  float val;
  float res = __acpp_sscp_modf_f32(x, &val);
  *y = static_cast<FloatT>(val);
  return res;
}

template<class FloatT>
HIPSYCL_BUILTIN double __acpp_modf(double x, FloatT* y) noexcept {
  double val;
  double res = __acpp_sscp_modf_f64(x, &val);
  *y = static_cast<FloatT>(val);
  return res;
}

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(nextafter)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(powr)

template<class IntType>
HIPSYCL_BUILTIN float __acpp_pown(float x, IntType y) noexcept {
  return __acpp_sscp_pown_f32(x, static_cast<__acpp_int32>(y));
}

template<class IntType>
HIPSYCL_BUILTIN double __acpp_pown(double x, IntType y) noexcept {
  return __acpp_sscp_pown_f64(x, static_cast<__acpp_int32>(y));
}

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN2(remainder)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(rint)


template<class IntType>
HIPSYCL_BUILTIN float __acpp_rootn(float x, IntType y) noexcept {
  return __acpp_sscp_rootn_f32(x, static_cast<__acpp_int32>(y));
}

template<class IntType>
HIPSYCL_BUILTIN double __acpp_rootn(double x, IntType y) noexcept {
  return __acpp_sscp_rootn_f64(x, static_cast<__acpp_int32>(y));
}

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(round)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(rsqrt)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(sqrt)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(sin)


template<class T, class FloatPtr>
HIPSYCL_BUILTIN T __acpp_sincos(T x, FloatPtr cosval) noexcept {
  *cosval = sscp_builtins::__acpp_cos(x);
  return sscp_builtins::__acpp_sin(x);
}

HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(sinh)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(sinpi)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(tan)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(tanh)
HIPSYCL_DEFINE_SSCP_GENFLOAT_MATH_BUILTIN(trunc)


// ***************** native math builtins ******************

#define HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(name)                      \
  HIPSYCL_BUILTIN float __acpp_native_##name(float x) {                     \
    return __acpp_sscp_native_##name##_f32(x);                              \
  }                                                                            \
  HIPSYCL_BUILTIN double __acpp_native_##name(double x) {                   \
    return __acpp_sscp_native_##name##_f64(x);                              \
  }

#define HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN2(name)                     \
  HIPSYCL_BUILTIN float __acpp_native_##name(float x, float y) {            \
    return __acpp_sscp_native_##name##_f32(x, y);                           \
  }                                                                            \
  HIPSYCL_BUILTIN double __acpp_native_##name(double x, double y) {         \
    return __acpp_sscp_native_##name##_f64(x, y);                           \
  }


HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(cos)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN2(divide)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(exp)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(exp2)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(exp10)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(log)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(log2)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(log10)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN2(powr)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(recip)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(rsqrt)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(sin)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(sqrt)
HIPSYCL_DEFINE_SSCP_GENFLOAT_NATIVE_BUILTIN(tan)

// ***************** half precision math builtins ******************


template<class T>
HIPSYCL_BUILTIN T __acpp_half_cos(T x) noexcept {
  return sscp_builtins::__acpp_cos(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_divide(T x, T y) noexcept {
  return x / y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp(T x) noexcept {
  return sscp_builtins::__acpp_exp(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp2(T x) noexcept {
  return sscp_builtins::__acpp_exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp10(T x) noexcept {
  return sscp_builtins::__acpp_exp10(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log(T x) noexcept {
  return sscp_builtins::__acpp_log(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log2(T x) noexcept {
  return sscp_builtins::__acpp_log2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log10(T x) noexcept {
  return sscp_builtins::__acpp_log10(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_powr(T x, T y) noexcept {
  return sscp_builtins::__acpp_powr(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_recip(T x) noexcept {
  return sscp_builtins::__acpp_native_divide(T{1}, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_rsqrt(T x) noexcept {
  return sscp_builtins::__acpp_rsqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_sin(T x) noexcept {
  return sscp_builtins::__acpp_sin(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_sqrt(T x) noexcept {
  return sscp_builtins::__acpp_sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_tan(T x) noexcept {
  return sscp_builtins::__acpp_tan(x);
}

// ***************** integer functions **************

template<class T, std::enable_if_t<std::is_signed_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_abs(T x) noexcept {
  return (x < 0) ? -x : x;
}

template<class T, std::enable_if_t<!std::is_signed_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_abs(T x) noexcept {
  return x;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_max(T x, T y) noexcept {
  return (x > y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_min(T x, T y) noexcept {
  return (x < y) ? x : y;
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  return sscp_builtins::__acpp_min(
    sscp_builtins::__acpp_max(x, minval), maxval);
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 1),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {
  return __acpp_sscp_ctz_u8(static_cast<__acpp_uint8>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 2),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {
  return __acpp_sscp_ctz_u16(static_cast<__acpp_uint16>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 4),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {
  return __acpp_sscp_ctz_u32(static_cast<__acpp_uint32>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 8),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {
  return __acpp_sscp_ctz_u64(static_cast<__acpp_uint64>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 1),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {
  return __acpp_sscp_clz_u8(static_cast<__acpp_uint8>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 2),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {
  return __acpp_sscp_clz_u16(static_cast<__acpp_uint16>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 4),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {
  return __acpp_sscp_clz_u32(static_cast<__acpp_uint32>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 8),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {
  return __acpp_sscp_clz_u64(static_cast<__acpp_uint64>(x));
}


template<class T, std::enable_if_t<std::is_signed_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_mul24(T x, T y) noexcept {
  return __acpp_sscp_mul24_s32(x, y);
}

template<class T, std::enable_if_t<!std::is_signed_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_mul24(T x, T y) noexcept {
  return __acpp_sscp_mul24_u32(x, y);
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) < 4),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_popcount(T x) noexcept {
  //we convert to the unsigned type to avoid the typecast creating
  //additional ones in front of the value if x is negative
  using Usigned = typename std::make_unsigned<T>::type;
  return __acpp_sscp_popcount_u32(static_cast<__acpp_uint32>(static_cast<Usigned>(x)));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 4),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_popcount(T x) noexcept {
  return __acpp_sscp_popcount_u32(static_cast<__acpp_uint32>(x));
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 8),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_popcount(T x) noexcept {
  return __acpp_sscp_popcount_u64(static_cast<__acpp_uint64>(x));
}

// **************** common functions *****************

template<class T, std::enable_if_t<!std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  return sscp_builtins::__acpp_fmin(
    sscp_builtins::__acpp_fmax(x, minval), maxval);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_degrees(T x) noexcept {
  return (180.f / M_PI) * x;
}

// __acpp_max() and __acpp_min() are handled by the overloads from the
// integer functions

template<class T>
HIPSYCL_BUILTIN T __acpp_mix(T x, T y, T a) noexcept {
  return x + (y - x) * a;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_radians(T x) noexcept {
  return (M_PI / 180.f) * x;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_step(T edge, T x) noexcept {
  return (x < edge) ? T{0} : T{1};
}

template<class T>
HIPSYCL_BUILTIN T __acpp_smoothstep(T edge0, T edge1, T x) noexcept {
  T t = sscp_builtins::__acpp_clamp((x - edge0) / (edge1 - edge0), T{0},
                                        T{1});
  return t * t * (3 - 2 * t);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sign_no_nan(T x) noexcept {
  return (x == T{0}) ? x : ((x > 0) ? T{1} : T{-1});
}


HIPSYCL_BUILTIN float __acpp_sign(float x) noexcept {
  if(__acpp_sscp_isnan_f32(x))
    return 0.0f;
  return __acpp_sign_no_nan(x);
}

HIPSYCL_BUILTIN double __acpp_sign(double x) noexcept {
  if(__acpp_sscp_isnan_f64(x))
    return 0.0;
  return __acpp_sign_no_nan(x);
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__acpp_cross3(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(),
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x()};
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__acpp_cross4(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(), 
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x(),
          typename VecType::element_type{0}};
}

// ****************** geometric functions ******************

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_dot(T a, T b) noexcept {
  return a * b;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __acpp_dot(T a, T b) noexcept {
  typename T::element_type result = 0;
  for (int i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __acpp_length(T a) noexcept {
  auto d = sscp_builtins::__acpp_dot(a, a);
  return sscp_builtins::__acpp_sqrt(d);
}

template<class T>
HIPSYCL_BUILTIN auto __acpp_distance(T a, T b) noexcept {
  return sscp_builtins::__acpp_length(a - b);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_normalize(T a) noexcept {
  // TODO rsqrt might be more efficient
  return a / sscp_builtins::__acpp_length(a);
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_fast_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __acpp_fast_length(T a) noexcept {
  auto d = sscp_builtins::__acpp_dot(a, a);
  return sscp_builtins::__acpp_half_sqrt(d);
}

template<class T>
HIPSYCL_BUILTIN auto __acpp_fast_distance(T a, T b) noexcept {
  return sscp_builtins::__acpp_fast_length(a - b);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fast_normalize(T a) noexcept {
  // TODO use rsqrt
  return a / sscp_builtins::__acpp_fast_length(a);
}

// ****************** relational functions ******************

#define HIPSYCL_DEFINE_SSCP_GENFLOAT_REL_BUILTIN(name)                         \
  HIPSYCL_BUILTIN int __acpp_##name(float x) {                              \
    return __acpp_sscp_##name##_f32(x);                                     \
  }                                                                            \
  HIPSYCL_BUILTIN int __acpp_##name(double x) {                             \
    return __acpp_sscp_##name##_f64(x);                                     \
  }

HIPSYCL_DEFINE_SSCP_GENFLOAT_REL_BUILTIN(isnan)

HIPSYCL_DEFINE_SSCP_GENFLOAT_REL_BUILTIN(isinf)

HIPSYCL_DEFINE_SSCP_GENFLOAT_REL_BUILTIN(isfinite)

HIPSYCL_DEFINE_SSCP_GENFLOAT_REL_BUILTIN(isnormal)

HIPSYCL_DEFINE_SSCP_GENFLOAT_REL_BUILTIN(signbit)

}
}
}

#endif

#endif
