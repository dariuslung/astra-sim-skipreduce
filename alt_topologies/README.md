# Alternate Topologies Wiki

This wiki documents the structural layout and physical mapping of the 9 configurations under `alt_topologies/`. These topologies allow testing SkipReduce under flat, direct-peer hierarchical, and switched hierarchical network layers.

---

## Topology Naming & Structure Conventions

Configurations follow the prefix naming convention of `L2-<L2>_L1-<L1>_N-<N>`, where:
* `N`: Total number of simulated endpoint nodes (NPUs).
* `L1`: Dimensions of the first hierarchy layer (group size).
* `L2`: Dimensions of the second hierarchy layer (hierarchical switch connections).

In the physical network maps (`*_network.txt`):
* Endpoint NPUs are assigned indices `0` through `N - 1`.
* Switch nodes are assigned indices starting from `N`.
* Links are defined with data rate (e.g., `100Gbps`), channel latency (e.g., `100ns`), and error rate.

---

## Visualizations & Mappings

### 1. `N-2`
* **Layout:** Flat FullyConnected/Switch configuration on 2 NPUs.
* **Nodes:** NPUs 0, 1; Switch 2.
* **Diagram:**
  ```text
               [ L1 Switch 2 ]
                //         \\
            [ NPU 0 ]     [ NPU 1 ]
  ```

---

### 2. `L1-1_N-2`
* **Layout:** Flat Switch configuration on 2 NPUs.
* **Nodes:** NPUs 0, 1; Switch 2.
* **Diagram:**
  ```text
               [ L1 Switch 2 ]
                //         \\
            [ NPU 0 ]     [ NPU 1 ]
  ```

---

### 3. `L1-1_N-4`
* **Layout:** Flat Switch configuration on 4 NPUs.
* **Nodes:** NPUs 0, 1, 2, 3; Switch 4.
* **Diagram:**
  ```text
                    [ L1 Switch 4 ]
                   //  ||     ||  \\
                 NPU  NPU     NPU  NPU
                  0    1       2    3
  ```

---

### 4. `L1-2_N-2`
* **Layout:** Hierarchical `[ Switch, FullyConnected ]` on 2 NPUs. Group size = 1. L1 switches directly connected via a peer link.
* **Nodes:** NPUs 0, 1; L1 Switches 2, 3.
* **Diagram:**
  ```text
               ============================= (Peer Link)
                ||                        ||
          [ L1 Switch 2 ]           [ L1 Switch 3 ]
                ||                        ||
             [ NPU 0 ]                 [ NPU 1 ]
  ```

---

### 5. `L1-2_N-4`
* **Layout:** Hierarchical `[ Switch, FullyConnected ]` on 4 NPUs. Group size = 2. L1 switches directly connected via a peer link.
* **Nodes:** NPUs 0, 1, 2, 3; L1 Switches 4, 5.
* **Diagram:**
  ```text
                     ============================= (Peer Link)
                      ||                        ||
               [ L1 Switch 4 ]           [ L1 Switch 5 ]
                //         \\             //         \\
            [ NPU 0 ]     [ NPU 1 ]   [ NPU 2 ]     [ NPU 3 ]
  ```

---

### 6. `L1-2_N-8`
* **Layout:** Hierarchical `[ Switch, FullyConnected ]` on 8 NPUs. Group size = 4. L1 switches directly connected via a peer link.
* **Nodes:** NPUs 0-7; L1 Switches 8, 9.
* **Diagram:**
  ```text
                       ============================= (Peer Link)
                        ||                        ||
                 [ L1 Switch 8 ]           [ L1 Switch 9 ]
                  //  ||  ||  \\            //  ||  ||  \\
                NPU  NPU NPU NPU          NPU  NPU NPU NPU
                 0    1   2   3            4    5   6   7
  ```

---

### 7. `L2-1_L1-2_N-2`
* **Layout:** Hierarchical `[ Switch, Switch ]` on 2 NPUs. Group size = 1. L1 switches connected via central L2 switch.
* **Nodes:** NPUs 0, 1; L1 Switches 2, 3; L2 Switch 4.
* **Diagram:**
  ```text
                    [ Central L2 Switch 4 ]
                      //              \\
                     //                \\
               [ L1 Switch 2 ]   [ L1 Switch 3 ]
                     ||                ||
                  [ NPU 0 ]         [ NPU 1 ]
  ```

---

### 8. `L2-1_L1-2_N-4`
* **Layout:** Hierarchical `[ Switch, Switch ]` on 4 NPUs. Group size = 2. L1 switches connected via central L2 switch.
* **Nodes:** NPUs 0, 1, 2, 3; L1 Switches 4, 5; L2 Switch 6.
* **Diagram:**
  ```text
                        [ Central L2 Switch 6 ]
                          //              \\
                         //                \\
                   [ L1 Switch 4 ]   [ L1 Switch 5 ]
                    //         \\     //         \\
                [ NPU 0 ]   [ NPU 1 ] [ NPU 2 ]   [ NPU 3 ]
  ```

---

### 9. `L2-1_L1-2_N-8`
* **Layout:** Hierarchical `[ Switch, Switch ]` on 8 NPUs. Group size = 4. L1 switches connected via central L2 switch.
* **Nodes:** NPUs 0-7; L1 Switches 8, 9; L2 Switch 10.
* **Diagram:**
  ```text
                              [ Central L2 Switch 10 ]
                                //                \\
                               //                  \\
                         [ L1 Switch 8 ]     [ L1 Switch 9 ]
                         //  ||  ||  \\       //  ||  ||  \\
                       NPU  NPU NPU NPU     NPU  NPU NPU NPU
                        0    1   2   3       4    5   6   7
  ```
