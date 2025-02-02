import argparse

import h5py


parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="Path to the computation file")
args = parser.parse_args()

with h5py.File(args.file, "r") as f:
    print("Attributes:")
    for a in f.attrs:
        print(f"\t {a}: {f.attrs[a]}")

    print("Datasets:")
    for d in f:
        print(d)
        print(f"\t Shape: {f[d].shape}")  # pyright: ignore
        for a in f[d].attrs:
            print(f"\t {a}: {f[d].attrs[a]}")
