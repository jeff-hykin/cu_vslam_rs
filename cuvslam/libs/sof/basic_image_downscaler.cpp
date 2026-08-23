
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

#include "sof/basic_image_downscaler.h"

namespace {

// the type was selected by measuring tracker performance on Jetson Nano.
// Minimal buffer_t capable to hold values is uint16_t. uint16_t is better then uint32_t and float.
using buffer_t = uint16_t;

buffer_t binom1(uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4, uint8_t v5) {
  const int v = v1 + v5 + 4 * (v2 + v4) + 6 * v3;
  assert(0 <= v && v <= 16 * 255);
  return static_cast<buffer_t>(v);
}

uint8_t binom2(buffer_t v1, buffer_t v2, buffer_t v3, buffer_t v4, buffer_t v5) {
  const int v = v1 + v5 + 4 * (v2 + v4) + 6 * v3;

  assert(0 <= v && v <= 256 * 255);
  // v <= 255 * 256
  // v + 128 <= 255 * 256 + 128
  // (v + 128) / 256 <= 255 + 128/256
  // 256 is the sum of weights
  // +128 to round up
  return static_cast<uint8_t>((v + 128) / 256);
}

void scale_down_w(const int h, const int iw, const int ow, const uint8_t *const idata, buffer_t *const odata) {
  assert(iw / 2 == ow);
  assert(idata != nullptr);
  assert(odata != nullptr);

  {
    const uint8_t *s = idata;
    buffer_t *t = odata;

    for (int y = 0; y < h; ++y) {
      // mirror image at left edge (between second and third items)
      // s[1] s[0] | s[0] s[1] s[2]
      *t = binom1(s[1], s[0], s[0], s[1], s[2]);

      s += iw;
      t += ow;
    }
  }

  for (int y = 0; y < h; ++y) {
    const uint8_t *s = idata + iw * y + 2;
    buffer_t *t = odata + ow * y + 1;

    for (int x = 1; x < ow - 1; ++x) {
      *t = binom1(s[-2], s[-1], s[0], s[1], s[2]);

      s += 2;
      ++t;
    }
  }

  {
    const uint8_t *s = idata + 2 * (ow - 1);
    buffer_t *t = odata + ow - 1;

    if (2 * ow == iw) {
      for (int y = 0; y < h; ++y) {
        // mirror image at right edge (between fourth and fifths items)
        // s[-2] s[-1] s[0] s[1] | s[1]
        *t = binom1(s[-2], s[-1], s[0], s[1], s[1]);

        s += iw;
        t += ow;
      }
    } else {
      for (int y = 0; y < h; ++y) {
        *t = binom1(s[-2], s[-1], s[0], s[1], s[2]);

        s += iw;
        t += ow;
      }
    }
  }

}  // scale_down_w

void scale_down_h(const int w, const int ih, const int oh, const buffer_t *const idata, uint8_t *const odata) {
  assert(ih / 2 == oh);
  assert(idata != nullptr);
  assert(odata != nullptr);

  const int w0 = -2 * w;
  const int w1 = -1 * w;
  const int w2 = 0 * w;
  const int w3 = 1 * w;
  const int w4 = 2 * w;

  {
    const buffer_t *s = idata;
    uint8_t *t = odata;
    for (int x = 0; x < w; ++x) {
      // mirror image at top edge (between second and third items)
      // s[w3] s[w2] | s[w2] s[w3] s[w4]
      *t = binom2(s[w3], s[w2], s[w2], s[w3], s[w4]);

      ++s;
      ++t;
    }
  }

  for (int y = 1; y < oh - 1; ++y) {
    const buffer_t *s = idata + w * 2 * y;
    uint8_t *t = odata + w * y;

    for (int x = 0; x < w; ++x) {
      *t = binom2(s[w0], s[w1], s[w2], s[w3], s[w4]);

      ++s;
      ++t;
    }
  }

  {
    const buffer_t *s = idata + w * 2 * (oh - 1);
    uint8_t *t = odata + w * (oh - 1);
    if (2 * oh == ih) {
      for (int x = 0; x < w; ++x) {
        // mirror image at bottom edge (between fourth and fifths items)
        // s[w0] s[w2] s[w2] s[w3] | s[w3]
        *t = binom2(s[w0], s[w1], s[w2], s[w3], s[w3]);

        ++s;
        ++t;
      }
    } else {
      for (int x = 0; x < w; ++x) {
        *t = binom2(s[w0], s[w1], s[w2], s[w3], s[w4]);

        ++s;
        ++t;
      }
    }
  }
}  // scale_down_h

}  // namespace

namespace cuvslam::sof {

void BasicImageDownscaler::compute_new_size(int &w, int &h) {
  w /= 2;
  h /= 2;
}

void BasicImageDownscaler::scale(const ImageSource &input, const ImageShape &ishape, ImageSource &output,
                                 const ImageShape &oshape) {
  assert(input.type == ImageSource::U8);
  assert(output.type == ImageSource::U8);
  assert(oshape.width == ishape.width / 2);
  assert(oshape.height == ishape.height / 2);

  assert(input.data != nullptr);
  assert(output.data != nullptr);

  const int iw = ishape.width;
  const int ih = ishape.height;

  const int ow = oshape.width;
  const int oh = oshape.height;

  const uint8_t *const idata = static_cast<const uint8_t *>(input.data);
  uint8_t *const odata = static_cast<uint8_t *>(output.data);

  buffer_.resize(ow * ih);
  buffer_t *const bdata = buffer_.data();

  // map iw * ih -> ow * ih
  scale_down_w(ih, iw, ow, idata, bdata);

  // map ow * ih -> ow * oh
  scale_down_h(ow, ih, oh, bdata, odata);
}

}  // namespace cuvslam::sof
