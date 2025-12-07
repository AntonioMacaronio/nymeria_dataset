import tyro
import os

def main(data_dir: str = "/data/nymeria"):
    print(f"Scanning {data_dir} for orphaned HDF5 files... (hdf5 files without corresponding mp4 files)")
    hdf5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    mp4_files = set([f for f in os.listdir(data_dir) if f.endswith(".mp4")])

    orphaned_hdf5_files = []
    for hdf5_file in hdf5_files:
        if hdf5_file.replace(".h5", ".mp4") not in mp4_files:
            print(f"Orphaned HDF5 file: {hdf5_file}")
            orphaned_hdf5_files.append(hdf5_file)
    
    print(f"Found {len(orphaned_hdf5_files)} orphaned HDF5 files")
    
    print("Would you like to delete the orphaned HDF5 files? (y/n)")
    answer = input()
    if answer == "y":
        for hdf5_file in orphaned_hdf5_files:
            os.remove(os.path.join(data_dir, hdf5_file))
            print(f"Deleted {hdf5_file}")
    else:
        print("Aborting...")

if __name__ == "__main__":
    tyro.cli(main)