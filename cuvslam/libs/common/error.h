
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

#pragma once

#include <cassert>

#define _CRT_STRINGIZE_(x) #x
#define _CRT_STRINGIZE(x) _CRT_STRINGIZE_(x)

namespace cuvslam {

class ErrorCode {
public:
  using ErrorNumericType = int;

  // ErrorCodes to return some of the common error results.
  // Since we take no dependency on Win32 API we need to come up with something similar and decision to use HRESULT
  // compatible error code model makes a lot of sense. Error codes for generic errors are taken from windows COM
  // HRESULTs errors for easy lookup during runtime failures as well as debugging. If more error codes are needed, it
  // can be added. Please follow the same model of error code format found in <winerror.h> under COM Error Codes section

#define _ERROR_ENUM_ITEM(name, value) name = ErrorNumericType(value)

  enum Enum : ErrorNumericType {
    _ERROR_ENUM_ITEM(S_True, 0),              // Operation completed Successfully
    _ERROR_ENUM_ITEM(S_False, 1),             // Operation resulted in No Error condition but no operation was performed
    _ERROR_ENUM_ITEM(E_Pending, 0x8000000A),  // The data necessary to complete this operation is not yet available
    _ERROR_ENUM_ITEM(E_Bounds, 0x8000000B),   // The operation attempted to access data outside the valid range
    _ERROR_ENUM_ITEM(E_Unexpected, 0x8000FFFF),  // Catastrophic failure due to unexpected condition (generally used as
                                                 // initialization for ErrorCode and if returned indicates coding error)
    _ERROR_ENUM_ITEM(E_NotImplemented, 0x80004001),  // Method not implemented
    _ERROR_ENUM_ITEM(E_Pointer, 0x80004003),         // Invalid pointer
    _ERROR_ENUM_ITEM(E_Abort, 0x80004004),           // Operation aborted
    _ERROR_ENUM_ITEM(E_Failure,
                     0x80004005),  // Unspecified error (generic error if it does not fall into any other category)
    _ERROR_ENUM_ITEM(E_FileNotFound, 0x80070002),    // File can't be found
    _ERROR_ENUM_ITEM(E_AccessDenied, 0x80070005),    // General access denied error (memory or other resource access)
    _ERROR_ENUM_ITEM(E_OutOfMemory, 0x8007000E),     // Failure due to memory allocation
    _ERROR_ENUM_ITEM(E_InvalidArgument, 0x80070057)  // One or more arguments are invalid
  };

  ErrorCode() : error_(E_Unexpected) {}
  ErrorCode(const Enum e) : error_(e) {}
  operator Enum() const { return error_; }
  operator ErrorNumericType() const { return ErrorNumericType(error_); }
  operator bool() const { return ErrorNumericType(error_) >= 0; }

  static const char* GetString(const Enum e) {
#define _ERROR_CASE_ITEM(name) \
  case name:                   \
    return _CRT_STRINGIZE(name)

    switch (e) {
      _ERROR_CASE_ITEM(S_True);
      _ERROR_CASE_ITEM(S_False);
      _ERROR_CASE_ITEM(E_Pending);
      _ERROR_CASE_ITEM(E_Bounds);
      _ERROR_CASE_ITEM(E_Unexpected);
      _ERROR_CASE_ITEM(E_NotImplemented);
      _ERROR_CASE_ITEM(E_Pointer);
      _ERROR_CASE_ITEM(E_Abort);
      _ERROR_CASE_ITEM(E_Failure);
      _ERROR_CASE_ITEM(E_FileNotFound);
      _ERROR_CASE_ITEM(E_AccessDenied);
      _ERROR_CASE_ITEM(E_OutOfMemory);
      _ERROR_CASE_ITEM(E_InvalidArgument);

      default:
        assert(false);
        return "Unknown ErrorCode";
    }
  }
  const char* str() const { return GetString(*this); }

private:
  Enum error_;
};

}  // namespace cuvslam
