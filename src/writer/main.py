import shutil
import argparse

import numpy as np
import fsspec
import zarr


def write_prediction(source_file, destination, endpoint):
    """Write contents from file to a s3 bucket."""
    print(f"Write new prediction to s3 ({endpoint})")
    fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})

    with open(source_file, "rb") as src:
        with fs.open(
            destination,
            mode="wb",
        ) as dest:
            print(f"write to destination: {dest}")
            dest.write(src.read())
    print(f"File {source_file} written to {destination}")


def load_zarr(source_folder, local_folder, endpoint):
    """Fetch zarr file from s3 and save it to the local filesystem."""

    if not source_folder.endswith("zarr"):
        print("Can only load zarr files from s3 to local filesystem")

    fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})
    print(fs)
    if not fs.exists(source_folder):
        print("Error: The zarr file doesn't exist in the bucket")
        return

    # Ensure local folder exists (remove if exists)
    shutil.rmtree(local_folder, ignore_errors=True)

    # Create local Zarr storee
    zarr_fs = zarr.DirectoryStore(local_folder)

    # Open the remote Zarr dataset
    s3_store = fs.get_mapper(source_folder)

    # Copy data from S3 to local
    zarr.copy_store(s3_store, zarr_fs, if_exists="replace")

    print(f"Zarr dataset downloaded from '{source_folder}' to '{local_folder}'")


def update_meta(dest, **kwargs):
    for index, value in kwargs.items():
        dest.attrs[index] = value


def concat(first, second):
    """Copy data and dates fields from first zarr and concatenate them to the second one."""
    src = zarr.open(first, mode="r")
    dest = zarr.open(second, mode="a")

    # Loop over all datasets in the first Zarr file
    for dataset_name in src:

        # Only interested of these two for now
        if dataset_name not in ("dates", "data"):
            continue

        # Concatenate along axis 0
        combined_data = np.concatenate(
            [src[dataset_name][:], dest[dataset_name][:]], axis=0
        )

        # Resize the dataset
        dest[dataset_name].resize(combined_data.shape)

        # Write the new data
        dest[dataset_name][:] = combined_data

        print(f"Concatenated dataset: {dataset_name}, new shape: {combined_data.shape}")

    # Set metadata
    update_meta(dest, frequency=6, start_date=src.attrs["start_date"])


def override_coords(coords_file, dest):
    """Override lat/lon coordinates in the zarr file with custom values from a .nc file."""
    import xarray as xr

    # Load coordinates from .nc file
    coords = xr.open_dataset(coords_file)

    lon = coords.lon.values
    lat = coords.lat.values

    # Open destination zarr file
    zarr_dest = zarr.open(dest, mode="r+")

    # Update latitudes and longitudes
    if "latitudes" in zarr_dest:
        zarr_dest["latitudes"][:] = lat
        print(f"Updated latitudes in {dest}")
    else:
        print("No 'latitudes' dataset found in the zarr file.")

    if "longitudes" in zarr_dest:
        zarr_dest["longitudes"][:] = lon
        print(f"Updated longitudes in {dest}")
    else:
        print("No 'longitudes' dataset found in the zarr file.")


def main():
    parser = argparse.ArgumentParser(
        description="Read Zarr files from S3, concatenate them by dates and write single files to S3"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="The source path (e.g., s3://bucket-name/path/to/file.txt)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="The destination path (e.g., s3://bucket-name/path/to/file.txt)",
    )
    parser.add_argument(
        "--coords",
        required=False,
        help="Override lat/lon with custom values from a .nc file",
    )
    parser.add_argument("--endpoint", default="https://lake.fmi.fi")
    args = parser.parse_args()

    if args.src.startswith("s3"):
        load_zarr(args.src, args.dest, args.endpoint)
    elif args.dest.startswith("s3"):
        write_prediction(args.src, args.dest, args.endpoint)
    elif "zarr" in args.src and "zarr" in args.dest:
        # two local files
        concat(args.src, args.dest)
    elif args.coords and "zarr" in args.dest:
        override_coords(args.coords, args.dest)


if __name__ == "__main__":
    main()
