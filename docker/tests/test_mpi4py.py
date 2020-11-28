#!/usr/bin/env python
# -*- coding: utf-8 -*-



import sys
import time
import pickle


#_______________________________________________________________________________
# Parse input arguments
#

test_bcast  = False
test_gather = False

if "bcast" in sys.argv:
    test_bcast = True

if "gather" in sys.argv:
    test_gather = True

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Time MPI initialization
#

t_start = time.time()

from mpi4py import MPI
comm     = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

t_init = time.time()
duration_init = t_init - t_start

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Time MPI Bcast
#

if mpi_rank == 0:
    payload = 1.2
else:
    payload = None

comm.Barrier()
t_start = time.time()

if test_bcast:
    payload = comm.bcast(payload, root=0)

t_bcast = time.time()
duration_bcast = t_bcast - t_start

# validate results
if test_bcast:
    if mpi_rank != 0:
        assert payload == 1.2

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Time MPI Gather
#

payload = 1.2

comm.Barrier()
t_start = time.time()

if test_gather:
    all_payloads = comm.gather(payload, root=0)

t_gather = time.time()
duration_gather = t_gather - t_start

# validate results
if test_gather:
    if mpi_rank == 0:
        for i in range(mpi_size):
            assert all_payloads[i] == payload

#-------------------------------------------------------------------------------



#_______________________________________________________________________________
# Report results
#

report = dict(rank=mpi_rank, size=mpi_size, init=duration_init)

if test_bcast:
    report["bcast"] = duration_bcast

if test_gather:
    report["gather"] = duration_gather

with open(f"report_{mpi_rank}.pkl", "wb") as f:
    pickle.dump(report, f)


MPI.Finalize()
