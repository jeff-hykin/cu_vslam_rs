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

#include "common/tga.h"

#include <cstring>
#include <fstream>

#include "common/log.h"

namespace cuvslam {

namespace {

// Helper to load TGA image
bool LoadTgaImpl(const std::string& filename, ImageMatrix<uint8_t>& image) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // TGA header is 18 bytes
  uint8_t header[18];
  file.read(reinterpret_cast<char*>(header), sizeof(header));

  if (header[2] != 3) {
    TraceError("Unsupported TGA image type. Only grayscale images (Type 3) are supported.");
    return false;
  }
  if (header[16] != 8) {
    TraceError("Unsupported TGA image type. Only 8 bit images are supported.");
    return false;
  }
  const uint8_t image_descriptor_bits = header[17];
  if (image_descriptor_bits != 0 && image_descriptor_bits != (1 << 5)) {
    TraceError("Unsupported TGA image type.");
    return false;
  }
  const bool origin_in_low_corner = (image_descriptor_bits & (1 << 5)) == 0;
  image.resize(header[14] + (header[15] << 8), header[12] + (header[13] << 8));

  file.read(reinterpret_cast<char*>(image.data()), image.rows() * image.cols());
  if (origin_in_low_corner) {
    image.colwise().reverseInPlace();
  }
  return file.good();
}

// Helper to save TGA image
bool SaveTgaImpl(const std::string& filename, const ImageMatrix<uint8_t>& image) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  if (!file.is_open()) {
    TraceError("Error opening file for writing: %s", filename.c_str());
    return false;
  }

  // TGA header is 18 bytes
  uint8_t header[18];
  std::memset(header, 0, sizeof(header));
  header[2] = 3;  // Image type (grayscale)
  header[12] = static_cast<uint8_t>(image.cols() & 0xFF);
  header[13] = static_cast<uint8_t>((image.cols() >> 8) & 0xFF);
  header[14] = static_cast<uint8_t>(image.rows() & 0xFF);
  header[15] = static_cast<uint8_t>((image.rows() >> 8) & 0xFF);
  header[16] = 8;     // Bits per pixel
  header[17] = 0x20;  // Bit 5 indicates an ordering of top-to-bottom

  file.write(reinterpret_cast<const char*>(header), sizeof(header));
  file.write(reinterpret_cast<const char*>(image.data()), image.rows() * image.cols());

  return file.good();
}

}  // namespace

bool LoadTga(const std::string& filename, ImageMatrix<uint8_t>& image) { return LoadTgaImpl(filename, image); }

bool SaveTga(const ImageMatrix<uint8_t>& image, const std::string& filename) { return SaveTgaImpl(filename, image); }

}  // namespace cuvslam
