
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

//////////////////////////////////////
// Statistical Variable helper classes
//////////////////////////////////////
#pragma once

#include <limits>       // for std::numeric_limits
#include <type_traits>  // for std::enable_if

#include "common/error.h"
#include "common/include_eigen.h"  // for Eigen::MatrixBase
#include "common/log.h"

namespace cuvslam {
template <typename _Type, typename _Enable = _Type>
struct _TypeTraits {};  // primary

template <typename _Type>
struct _TypeTraits<_Type, typename std::enable_if<std::is_arithmetic<_Type>::value, _Type>::type> {
  using Scalar = _Type;
  static _Type Zero() { return 0; }
  static _Type MinValue() { return std::numeric_limits<Scalar>::min(); }
  static _Type MaxValue() { return std::numeric_limits<Scalar>::max(); }
  static _Type Pow2(const _Type& x) { return x * x; }
  static _Type Min(const _Type& a, const _Type& b) { return std::min(a, b); }
  static _Type Max(const _Type& a, const _Type& b) { return std::max(a, b); }
};

template <typename _Type>
struct _TypeTraits<_Type,
                   typename std::enable_if<std::is_base_of<Eigen::MatrixBase<_Type>, _Type>::value, _Type>::type> {
  using Scalar = typename _Type::Scalar;
  static _Type Zero() { return _Type::Zero(); }
  static _Type MinValue() { return _Type::Constant(std::numeric_limits<Scalar>::min()); }
  static _Type MaxValue() { return _Type::Constant(std::numeric_limits<Scalar>::max()); }
  static _Type Pow2(const _Type& x) { return x.cwiseAbs2(); }
  static _Type Min(const _Type& a, const _Type& b) { return a.cwiseMin(b); }
  static _Type Max(const _Type& a, const _Type& b) { return a.cwiseMax(b); }
};

template <typename _Type, typename _Traits = void>
struct _AddonEmpty {
  void operator()(const _Type&) {}
};

template <typename _Type, typename _Traits = _TypeTraits<_Type>>
class _AddonMinMax {
  _Type min_ = _Traits::MaxValue();
  _Type max_ = _Traits::MinValue();

public:
  _AddonMinMax() = default;

  _Type min() const { return min_; }
  _Type max() const { return max_; }

  void operator()(const _Type& value) {
    min_ = _Traits::Min(min_, value);
    max_ = _Traits::Max(max_, value);
  }
};

template <typename _Type, typename _Addon = _AddonEmpty<_Type>, typename _Traits = _TypeTraits<_Type>>
class StatisticalBase : public _Addon {
  using Scalar = typename _Traits::Scalar;
  _Type mean_ = _Traits::Zero();
  _Type nvariance_ = _Traits::Zero();  // to get true variance do nvariance_ / count_
  size_t count_ = 0;

public:
  StatisticalBase() = default;

  _Type mean() const { return mean_; }
  _Type variance() const { return nvariance_ / Scalar(count_); }

  // Standard deviation
  _Type std_dev() const { return std::sqrt(variance()); }

  size_t count() const { return count_; }

  void operator()(const _Type& value) {
    _Addon::operator()(value);

    count_++;
    // use one pass formula http://suave_skola.varak.net/proj2/stddev.pdf
    const _Type delta = value - mean_;
    mean_ += delta / Scalar(count_);
    nvariance_ += _Traits::Pow2(delta) * Scalar(count_ - 1) / Scalar(count_);
  }
};

template <typename _Scalar, bool _PrintInDtor = false>
class NamedStatisticalVariable : public StatisticalBase<_Scalar, _AddonMinMax<_Scalar>> {
  const std::string name_;
  using Base = StatisticalBase<_Scalar, _AddonMinMax<_Scalar>>;

public:
  NamedStatisticalVariable() : Base(), name_("NO_NAME") {}
  NamedStatisticalVariable(const std::string& name) : Base(), name_(name) {}
  ~NamedStatisticalVariable() {
    TracePrintIf(_PrintInDtor, "%-80s = %-12g [SD = %-12g Max = %-12g Min = %-12g Samples = %zu]\n", name_.c_str(),
                 Base::mean(), std::sqrt(Base::variance()), Base::max(), Base::min(), Base::count());
  }
};

template <typename _Scalar, typename _NameType, bool _PrintInDtor, typename _StatScalar = double>
class PersistentNamedStatisticalVariable {
public:
  PersistentNamedStatisticalVariable() = default;
  PersistentNamedStatisticalVariable(const _Scalar& value) : value_(value) { set(value); }
  const _Scalar& operator=(const _Scalar& value) { return (value_ = set(value)); }
  PersistentNamedStatisticalVariable(const PersistentNamedStatisticalVariable& value) = default;
  PersistentNamedStatisticalVariable& operator=(const PersistentNamedStatisticalVariable&) = default;
  operator const _Scalar&() const { return value_; }

private:
  _Scalar value_;

  const _Scalar& set(const _Scalar& value) {
    static NamedStatisticalVariable<_StatScalar, _PrintInDtor> sv(_NameType::name());
    sv(value);
    return value;
  }

  template <typename _B, typename _N, bool _P, typename _S>
  PersistentNamedStatisticalVariable(const PersistentNamedStatisticalVariable<_B, _N, _P, _S>&) = delete;

  template <typename _B, typename _N, bool _P, typename _S>
  PersistentNamedStatisticalVariable& operator=(const PersistentNamedStatisticalVariable<_B, _N, _P, _S>&) = delete;
};

#define GTestStatVarT(Name)                                                                                       \
  struct _##Name {                                                                                                \
    static std::string name() {                                                                                   \
      return std::string(test_info_->test_case_name()) + "::" + test_info_->name() + "::" + _CRT_STRINGIZE(Name); \
    }                                                                                                             \
  };                                                                                                              \
  PersistentNamedStatisticalVariable<float, _##Name, true> Name

#define PersistStatVarT(Name)                                                                     \
  struct _##Name {                                                                                \
    static std::string name() { return std::string(__FUNCTION__) + "::" + _CRT_STRINGIZE(Name); } \
  };                                                                                              \
  PersistentNamedStatisticalVariable<float, _##Name, true> Name

template <typename _Type>
using Statistical = StatisticalBase<_Type>;
template <typename _Type>
using StatisticalLimits = StatisticalBase<_Type, _AddonMinMax<_Type>>;

}  // namespace cuvslam
