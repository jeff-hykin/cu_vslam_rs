# Load map under the hood (Mermaid)

Diagrams referenced from [load_map.md](load_map.md).

## System overview

```mermaid
flowchart LR
  subgraph RT["SLAM"]
    direction TB
    SM[SaveMap]
    LC[Localize]
    TR["Track (realtime)"]
  end

  subgraph Q["ThreadSafeMessageQueue"]
    direction LR
    m1[AddKF] --- m2[FindLC] --- m3[AddKF] --- m4[SaveMap] --- m5["..."] --- m6[FindLC] --- m7[Localize] --- m8[AddKF]
  end

  subgraph BG["SLAM"]
    direction TB
    BT["BackgroundThread (non-realtime)<br/>while true: process_message"]
    MAP[(map)]
    BT --- MAP
  end

  TAIL[(ThreadSafe Tail)]

  RT --> Q --> BG
  TR <--> TAIL
  TAIL <--> BT
```

## ThreadSafe tail (keyframe chain)

```mermaid
flowchart LR
  subgraph TAIL["ThreadSafe Tail (keyframe chain, time →)"]
    direction LR
    KF1((KF1)) --> KF2((KF2)) --> KF3((KF3)) --> KF4((KF4)) --> KF5((KF5)) --> KF6((KF6)) --> KF7((KF7)) --> KF8((KF8))
  end

  L1["Localized in map pose<br/>near the prior"]
  L2["LC if found<br/>already applied"]
  L3["Current KF processing<br/>in background thread"]
  L4["In queue for AddKF<br/>and FindLC"]
  L5["Current SlamPose"]

  L1 -.-> KF1
  L2 -.-> KF2
  L2 -.-> KF3
  L3 -.-> KF4
  L4 -.-> KF5
  L4 -.-> KF6
  L4 -.-> KF7
  L5 -.-> KF8
```

---

Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
