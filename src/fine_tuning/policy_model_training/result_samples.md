# Speech-to-Text Refiner Evaluation Samples

## Sample 9

> [!quote] **Raw ASR Input**
> the nccl all reduce ring topology is supposed to ensure optimal bandwidth utilization but we are seeing stalls in the transact log during heavy load if a process segfaults the nvidia driver doesn't always clean up the state and we have to manually terminat the zombie processes before we can allocat the gpu again the pcie switch topology on these dense nodes means that peer to peer connecti between certain gpu pairs is much slower than others affecting the nccl ring formation the rdma read and write queues need deep buffers to handle the bursty traffic from the distributed optimizer otherwise we get packet loss on the connecti during the distributed checkpointing phase all nodes try to write to the shared nfs simultaneously causing the storage throughput to crash and tasks to fail to allocat space the mig partitions are supposed to be hardware isolated but we've seen edge cases where high mem utilization on one slice affects the memory bandwidth of the others the synchronization overhead in the parameter server architecture was too high so moving to nccl all reduce was necessary but it introduced new complexities in how we configur the network

### Refined Output

The NCCL all-reduce ring topology is supposed to ensure optimal bandwidth utilization, but we are seeing stalls in the transaction log during heavy loads if a process segfaults the NVIDIA driver doesn't always clean up the state, and we have to manually terminate the zombie processes before we can allocate the GPU again.

---

The PCIe switch topology on these dense nodes means that peer-to-peer connections between certain GPU pairs are much slower than others, affecting the NCCL ring formation. The RDMA read and write queues need deep buffers to handle the bursty traffic from the distributed optimizer, otherwise, we get packet loss on the connection during the distributed checkpointing phase.

---

All nodes try to write to the shared NFS simultaneously, causing the storage throughput to crash and tasks to fail to allocate space. The MIG partitions are supposed to be hardware isolated, but we've seen edge cases where high memory utilization on one slice affects the memory bandwidth of the others.

---

The synchronization overhead in the parameter server architecture was too high, so moving to NCCL all-reduce was necessary, but it introduced new complexities in how we configure the network.

---

## Sample 10

> [!quote] **Raw ASR Input**
> um so basically before applying the foreign key migrat make sure all orphan records in the child table are purged otherwise the add constraint statement will abort with foreign key violation error

### Refined Output

Before applying the foreign key migration, make sure all orphan records in the child table are purged, otherwise the `ADD CONSTRAINT` statement will abort with a foreign key violation error.

---
