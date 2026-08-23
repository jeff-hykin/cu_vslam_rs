
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

#include <stdio.h>

#include "lmdb.h"

#include "common/include_gtest.h"

TEST(LmdbTest, TestAll) {
  // sample code from:
  // https://github.com/rvagg/archived-lmdb/blob/master/deps/liblmdb-20130601/sample-mdb.c

  int rc;
  MDB_env *env;
  MDB_dbi dbi;
  MDB_val key, data;
  MDB_txn *txn;
  MDB_cursor *cursor;
  char sval[32];

  rc = mdb_env_create(&env);
  rc = mdb_env_open(env, "./testdb", 0, 0664);
  if (rc) return;
  rc = mdb_txn_begin(env, NULL, 0, &txn);
  rc = mdb_open(txn, NULL, 0, &dbi);

  key.mv_size = sizeof(int);
  key.mv_data = sval;
  data.mv_size = sizeof(sval);
  data.mv_data = sval;

  sprintf(sval, "%03x %d foo bar", 32, 3141592);
  rc = mdb_put(txn, dbi, &key, &data, 0);
  rc = mdb_txn_commit(txn);
  if (rc) {
    fprintf(stderr, "mdb_txn_commit: (%d) %s\n", rc, mdb_strerror(rc));
    goto leave;
  }
  rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
  rc = mdb_cursor_open(txn, dbi, &cursor);
  while ((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0) {
    printf("key: %p %.*s, data: %p %.*s\n", key.mv_data, (int)key.mv_size, (char *)key.mv_data, data.mv_data,
           (int)data.mv_size, (char *)data.mv_data);
  }
  mdb_cursor_close(cursor);
  mdb_txn_abort(txn);
leave:
  mdb_close(env, dbi);
  mdb_env_close(env);
}
