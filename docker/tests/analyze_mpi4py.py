#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import glob
import pickle
import statistics as stat



def load_reports(dest="."):

    data = list()
    for name in glob.glob(os.path.join(dest, "report_*.pkl")):

        with open(name, "rb") as f:
            report = pickle.load(f)

        data.append(report)

    data.sort(key=lambda x:x["rank"])

    return data



def slice(data, key):
    ret = [0.]*len(data)

    for elt in data:
        ret[elt["rank"]] = elt[key]

    return ret



def stats(data, key):
    ar = slice(data, key)

    return min(ar), max(ar), stat.mean(ar), stat.stdev(ar)



has = lambda data, key: key in data[0] if len(data) > 0 else False



if __name__ == "__main__":

    data = load_reports()

    # init is always present
    s0, s1, s2, s3 = stats(data, "init")

    n_ranks = len(data)

    print(f"+------------------------------------------------------------+")
    print(f"|    MPI4PY benchmark results for {n_ranks:<13} ranks        |")
    print(f"+-----------+-----------+-----------+-----------+------------+")
    print(f"|           | min       | max       | mean      | stdev      |")
    print(f"+-----------+-----------+-----------+-----------+------------+")
    print(f"| init      | {s0:9.4f} | {s1:9.4f} | {s2:9.4f} | {s3:9.4f}  |")
    print(f"+-----------+-----------+-----------+-----------+------------+")

    if has(data, "bcast"):
        b0, b1, b2, b3 = stats(data, "bcast")
        print(f"| bcast     | {b0:9.4f} | {b1:9.4f} | {b2:9.4f} | {b3:9.4f}  |")
        print(f"+-----------+-----------+-----------+-----------+------------+")

    if has(data, "gather"):
        g0, g1, g2, g3 = stats(data, "gather")
        print(f"| gather    | {g0:9.4f} | {g1:9.4f} | {g2:9.4f} | {g3:9.4f}  |")
        print(f"+-----------+-----------+-----------+-----------+------------+")


