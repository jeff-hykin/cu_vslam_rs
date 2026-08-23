
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

#include "common/vector_3t.h"

namespace cuvslam::slam::grid {

using HashCellId = uint64_t;

// no direction
// #define BITS_PER_AXIS 21

// direction. 6 planes only
// #define BITS_PER_AXIS 20
// #define GRID_CELL_HAS_PLANE

// direction. 6 planes only. 2x2 subdivision
#define BITS_PER_AXIS 19
#define GRID_CELL_HAS_PLANE
#define GRID_CELL_PLANE_BITS_SUBDIV 1

struct CellId {
  // cell coords
  uint32_t x_ : BITS_PER_AXIS;
  uint32_t y_ : BITS_PER_AXIS;
  uint32_t z_ : BITS_PER_AXIS;
#ifdef GRID_CELL_HAS_PLANE
  uint32_t plane_ : 3;
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
  uint32_t u_ : GRID_CELL_PLANE_BITS_SUBDIV;
  uint32_t v_ : GRID_CELL_PLANE_BITS_SUBDIV;
#endif
#endif
  static const uint32_t zero_ = 1 << (BITS_PER_AXIS - 1);  // BITS_PER_AXIS-1

  CellId() {}
  CellId(int x, int y, int z) {
    x_ = x + zero_;
    y_ = y + zero_;
    z_ = z + zero_;
  }
  bool operator==(const CellId& arg) { return (x_ == arg.x_ && y_ == arg.y_ && z_ == arg.z_); }
  bool operator!=(const CellId& arg) { return (x_ != arg.x_ || y_ != arg.y_ || z_ != arg.z_); }

#ifndef GRID_CELL_HAS_PLANE
  static inline CellId CellIdFromXYZ(const Vector3T& eye, const Vector3T&, float cell_size) {
    CellId id;
    id.x_ = CellId::zero_ + (int)floor(double(eye.x()) / cell_size);
    id.y_ = CellId::zero_ + (int)floor(double(eye.y()) / cell_size);
    id.z_ = CellId::zero_ + (int)floor(double(eye.z()) / cell_size);
    return id;
  }
#else
  static inline CellId CellIdFromXYZ(const Vector3T& eye, const Vector3T& landmark, float cell_size) {
    CellId id;
    id.x_ = CellId::zero_ + (int)floor(double(eye.x()) / cell_size);
    id.y_ = CellId::zero_ + (int)floor(double(eye.y()) / cell_size);
    id.z_ = CellId::zero_ + (int)floor(double(eye.z()) / cell_size);

    // direction
    Vector3T dir = landmark - eye;
    dir.normalize();
    // detect max component of direction
    int plane = 0;  // X
    Vector2T uv(dir.y(), dir.z());
    if (std::abs(dir.y()) > std::abs(dir.x())) {
      plane = 1;  // Y
      uv = Vector2T(dir.x(), dir.z());
      if (std::abs(dir.z()) > std::abs(dir.y())) {
        plane = 2;  // Z
        uv = Vector2T(dir.x(), dir.y());
      }
    } else {
      if (std::abs(dir.z()) > std::abs(dir.x())) {
        plane = 2;  // Z
        uv = Vector2T(dir.x(), dir.y());
      }
    }
    // sign of selected component
    if (dir[plane] < 0) {
      plane += 3;
    }
    id.plane_ = plane;

#ifdef GRID_CELL_PLANE_BITS_SUBDIV
    // uv to [0, 1] band
    uv = uv * 0.5f + Vector2T(0.5f, 0.5f);
    // uv to int band
    int cells_per_plane = (1 << GRID_CELL_PLANE_BITS_SUBDIV);
    uv = uv * cells_per_plane;
    id.u_ = std::min(static_cast<int>(std::floor(uv.x())), cells_per_plane - 1);
    id.v_ = std::min(static_cast<int>(std::floor(uv.y())), cells_per_plane - 1);
#endif
    return id;
  }
#endif
  static inline void CellIdFromEye(const Vector3T& eye, std::vector<CellId>& cell_ids, float cell_size) {
#ifdef GRID_CELL_HAS_PLANE
    int cells_per_plane = 1;
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
    cells_per_plane = (1 << GRID_CELL_PLANE_BITS_SUBDIV);
#endif
    CellId id = CellIdFromXYZ(eye, eye, cell_size);
    cell_ids.resize(6 * cells_per_plane * cells_per_plane);
    int index = 0;
    for (int i = 0; i < 6; i++) {
      id.plane_ = i;
      for (int u = 0; u < cells_per_plane; u++) {
        for (int v = 0; v < cells_per_plane; v++) {
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
          id.u_ = u;
          id.v_ = v;
#endif
          cell_ids[index++] = id;
        }
      }
    }
#else
    const CellId id = CellIdFromXYZ(eye, eye, cell_size);
    cell_ids.resize(1);
    cell_ids[0] = id;
#endif
  }
  static inline HashCellId HashCellIdFromCellId(CellId id) {
    HashCellId hid = 0;
    hid |= id.x_;
    hid |= (uint64_t(id.y_) << (1 * BITS_PER_AXIS));
    hid |= (uint64_t(id.z_) << (2 * BITS_PER_AXIS));

#ifdef GRID_CELL_HAS_PLANE
    hid |= (uint64_t(id.plane_) << (3 * BITS_PER_AXIS));
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
    hid |= (uint64_t(id.u_) << (3 * BITS_PER_AXIS + 3));
    hid |= (uint64_t(id.v_) << (3 * BITS_PER_AXIS + 3 + GRID_CELL_PLANE_BITS_SUBDIV));
#endif
#endif
    return hid;
  }
  static inline CellId HashCellIdToCellId(HashCellId hash_cell_id) {
    CellId id;
    uint64_t mask = (1 << BITS_PER_AXIS) - 1;
    id.x_ = hash_cell_id & mask;
    hash_cell_id >>= BITS_PER_AXIS;
    id.y_ = hash_cell_id & mask;
    hash_cell_id >>= BITS_PER_AXIS;
    id.z_ = hash_cell_id & mask;
    hash_cell_id >>= BITS_PER_AXIS;

#ifdef GRID_CELL_HAS_PLANE
    id.plane_ = hash_cell_id & 0x7;
    hash_cell_id >>= 3;

#ifdef GRID_CELL_PLANE_BITS_SUBDIV
    uint64_t mask_uv = (1 << GRID_CELL_PLANE_BITS_SUBDIV) - 1;
    id.u_ = hash_cell_id & mask_uv;
    hash_cell_id >>= GRID_CELL_PLANE_BITS_SUBDIV;
    id.v_ = hash_cell_id & mask_uv;
    hash_cell_id >>= GRID_CELL_PLANE_BITS_SUBDIV;
#endif

#endif
    return id;
  }
  static inline HashCellId HashCellIdFromXYZ(const Vector3T& eye, const Vector3T& landmark, float cell_size) {
    CellId id = CellIdFromXYZ(eye, landmark, cell_size);
    return HashCellIdFromCellId(id);
  }
  static inline void HashCellIdFromEye(const Vector3T& eye, std::vector<HashCellId>& hashcell_ids, float cell_size) {
#ifdef GRID_CELL_HAS_PLANE
    int cells_per_plane = 1;
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
    cells_per_plane = (1 << GRID_CELL_PLANE_BITS_SUBDIV);
#endif
    CellId id = CellIdFromXYZ(eye, eye, cell_size);
    hashcell_ids.resize(6 * cells_per_plane * cells_per_plane);
    int index = 0;
    for (int i = 0; i < 6; i++) {
      id.plane_ = i;
      for (int u = 0; u < cells_per_plane; u++) {
        for (int v = 0; v < cells_per_plane; v++) {
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
          id.u_ = u;
          id.v_ = v;
#endif
          hashcell_ids[index++] = HashCellIdFromCellId(id);
        }
      }
    }
#else
    hashcell_ids.resize(1);
    CellId id = CellIdFromXYZ(eye, eye, cell_size);
    hashcell_ids[0] = HashCellIdFromCellId(id);
#endif
  }
  static inline Vector3T CellIdToXYZ(const CellId& id, float cell_size) {
    Vector3T xyz;
    xyz.x() = ((int64_t)id.x_ - CellId::zero_) * cell_size + cell_size * 0.5f;
    xyz.y() = ((int64_t)id.y_ - CellId::zero_) * cell_size + cell_size * 0.5f;
    xyz.z() = ((int64_t)id.z_ - CellId::zero_) * cell_size + cell_size * 0.5f;
    return xyz;
  }
  static inline Vector3T CellIdToDirection(const CellId& id) {
    Vector3T dir = Vector3T::UnitX();  // default
#ifdef GRID_CELL_HAS_PLANE
    int plane = id.plane_ % 3;
    if (plane == 1) {
      dir = Vector3T::UnitY();
    } else if (plane == 2) {
      dir = Vector3T::UnitZ();
    }
    bool inv = id.plane_ >= 3;
    if (inv) {
      dir = -dir;
    }
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
//        ss << "{" << cell_id.u_ << "|" << cell_id.v_ << "}";
#endif
#endif
    return dir;
  }
  static inline std::string CellIdToString(const CellId& cell_id) {
    std::stringstream ss;
    ss << "[" << (int64_t)cell_id.x_ - CellId::zero_ << "," << (int64_t)cell_id.y_ - CellId::zero_ << ","
       << (int64_t)cell_id.z_ - CellId::zero_ << "]";
#ifdef GRID_CELL_HAS_PLANE
    ss << "P" << cell_id.plane_;
#ifdef GRID_CELL_PLANE_BITS_SUBDIV
    ss << "{" << cell_id.u_ << "|" << cell_id.v_ << "}";
#endif

#endif
    return ss.str();
  }
};

}  // namespace cuvslam::slam::grid
