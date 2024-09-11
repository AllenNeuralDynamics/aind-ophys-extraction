import argparse
import json
import logging
import os
from datetime import datetime as dt
from datetime import timezone as tz
from pathlib import Path
from typing import Union, Tuple

import h5py
import numpy as np
import skimage
import suite2p
from aind_data_schema.core.processing import (
    DataProcess,
    PipelineProcess,
    Processing,
    ProcessName,
)


def get_r_from_min_mi(raw_trace, neuropil_trace, resolution=0.01, r_test_range=[0, 2]):
    """
    Get the r value that minimizes the mutual information between
    the corrected trace and the neuropil trace.

    Parameters
    ----------
    raw_trace : np.ndarray
        1D array of raw trace values.
    neuropil_trace : np.ndarray
        1D array of neuropil trace values.
    resolution : float
        Resolution of r values to test.
    r_test_range : list of float
        List of two floats representing the inclusive range of r values to test.

    Returns
    -------
    r_best : float
        r value that minimizes the mutual information between
        the corrected trace and the neuropil trace.
    mi_iters : np.ndarray
        1D array of mutual information values for each r value tested.
    r_iters : np.ndarray
        1D array of r values tested.
    """
    r_iters = np.arange(r_test_range[0], r_test_range[1] + resolution, resolution)
    mi_iters = np.zeros(len(r_iters))
    neuropil_trace[np.isnan(neuropil_trace)] = 0
    raw_trace[np.isnan(raw_trace)] = 0
    for r_i, r_temp in enumerate(r_iters):
        Fc = raw_trace - r_temp * neuropil_trace
        mi_iters[r_i] = skimage.metrics.normalized_mutual_information(
            Fc, neuropil_trace
        )
    min_ind = np.argmin(mi_iters)
    r_best = r_iters[min_ind]
    return r_best, mi_iters, r_iters


def get_FC_from_r(raw_trace, neuropil_trace, min_r_count=5):
    """
    Get the corrected trace from the raw trace and neuropil trace using the given r values.

    Parameters
    ----------
    raw_trace : np.ndarray
        1D array of raw trace values.
    neuropil_trace : np.ndarray
        1D array of neuropil trace values.
    min_r_count : int
        Minimum number of r values to use for mean r value calculation.

    Returns
    -------
    FCs : np.ndarray
        1D array of corrected traces for each r value.
    r_values : np.ndarray
        1D array of r values used for the correction.
    raw_r : np.ndarray
        1D array of r values that minimized the mutual information.
    """
    r_values = np.zeros(raw_trace.shape[0])
    FCs = np.zeros_like(raw_trace)
    for roi in range(raw_trace.shape[0]):
        r_values[roi], _, _ = get_r_from_min_mi(raw_trace[roi], neuropil_trace[roi])
    mean_r = np.mean(r_values[r_values < 1])
    if len(np.where(r_values < 1)[0]) < min_r_count:
        mean_r = 0.8
    raw_r = r_values.copy()
    r_values[r_values >= 1] = mean_r
    for roi in range(raw_trace.shape[0]):
        FCs[roi] = raw_trace[roi] - r_values[roi] * neuropil_trace[roi]
    return FCs, r_values, raw_r


def make_output_directory(output_dir: Path, experiment_id: str) -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    experiment_id: str
        experiment_id number

    Returns
    -------
    output_dir: str
        output directory
    """
    output_dir = output_dir / experiment_id
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / "extraction"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def write_output_metadata(
    metadata: dict,
    process_json_dir: str,
    process_name: str,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    start_date_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p cellpose segmentation
    input_fp: str
        path to data input
    output_fp: str
        path to data output
    """
    with open(Path(process_json_dir) / "processing.json", "r") as f:
        proc_data = json.load(f)
    processing = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="Multplane Ophys Processing Pipeline",
            pipeline_url="https://codeocean.allenneuraldynamics.org/capsule/7026342/tree",
            pipeline_version="0.5.0",
            data_processes=[
                DataProcess(
                    name=process_name,
                    software_version="1ce0d71d0ea74ad6c9f3b93c4db400696d885b4b",  # TODO: FIX THIS!!
                    start_date_time=start_date_time,
                    end_date_time=dt.now(tz.utc),
                    input_location=str(input_fp),
                    output_location=str(output_fp),
                    code_url=(os.getenv("EXTRACTION_URL")),
                    parameters=metadata,
                )
            ],
        )
    )
    prev_processing = Processing(**proc_data)
    prev_processing.processing_pipeline.data_processes.append(
        processing.processing_pipeline.data_processes[0]
    )
    prev_processing.write_standard_file(output_directory=Path(output_fp).parent)


def create_virtual_dataset(
    h5_file: Path, frame_locations: list, frames_length: int, temp_dir: Path
) -> Path:
    """Creates a virtual dataset from a list of frame locations

    Parameters
    ----------
    h5_file: Path
        path to h5 file
    frame_locations: list
        list of frame locations
    frames_length: int
        sum of frame locations
    temp_dir: Path
        temporary directory for virtual dataset

    Returns
    -------
    h5_file: Path
        path to virtual dataset
    """
    with h5py.File(h5_file, "r") as f:
        data_shape = f["data"].shape
        dtype = f["data"].dtype
        vsource = h5py.VirtualSource(f["data"])
        layout = h5py.VirtualLayout(shape=(frames_length, *data_shape[1:]), dtype=dtype)
        start = 0
        for loc in frame_locations:
            layout[start : start + loc[1] - loc[0] + 1] = vsource[loc[0] : loc[1] + 1]
            start += loc[1] - loc[0] + 1
        h5_file = temp_dir / h5_file.name
        with h5py.File(h5_file, "w") as f:
            f.create_virtual_dataset("data", layout)

    return h5_file


def bergamo_segmentation(motion_corr_fp: Path, session: dict, temp_dir: Path) -> str:
    """Performs singleplane motion correction on a singleplane data set

    Parameters
    ----------
    motion_corr_fp: Path
        path to data directory
    session: dict
        session information
    temp_dir: Path
        temporary directory for virtual dataset
    Returns
    -------
    h5_file: str
        path to motion corrected h5 file
    """
    motion_dir = motion_corr_fp.parent
    tiff_stem_locs = next(motion_dir.glob("tiff_stem_locations.json"))
    with open(tiff_stem_locs, "r") as j:
        tiff_stem_locations = json.load(j)
    valid_epoch_stems = [
        i["output_parameters"]["tiff_stem"]
        for i in session["stimulus_epochs"]
        if i["stimulus_name"] != "2p photostimulation"
    ]
    frame_locations = [tiff_stem_locations[i] for i in valid_epoch_stems]
    frames_length = sum([(i[1] - i[0] + 1) for i in frame_locations])

    return create_virtual_dataset(
        motion_corr_fp, frame_locations, frames_length, temp_dir
    )


def get_metdata(input_dir: Path) -> Tuple[dict, dict, dict]:
    """Get the session and data description metadata from the input directory

    Parameters
    ----------
    input_dir: Path
        input directory

    Returns
    -------
    session: dict
        session metadata
    data_description: dict
        data description metadata
    processing: dict

    """
    try:
        session_fp = next(input_dir.rglob("session.json"))
        with open(session_fp, "r") as j:
            session = json.load(j)
    except StopIteration:
        session = None
    try:
        data_des_fp = next(input_dir.rglob("data_description.json"))
        with open(data_des_fp, "r") as j:
            data_description = json.load(j)
    except StopIteration:
        data_description = None
    process_fp = next(input_dir.rglob("*/processing.json"))
    with open(process_fp, "r") as j:
        processing = json.load(j)

    return session, data_description, processing


def get_frame_rate(processing: dict) -> float:
    """Get the frame rate from the processing metadata

    Parameters
    ----------
    processing: dict
        processing metadata

    Returns
    -------
    frame_rate: float
        frame rate
    """
    if processing.get("processing_pipeline") is not None:
        processing = processing["processing_pipeline"]
    for data_proc in processing["data_processes"]:
        if data_proc["parameters"].get("movie_frame_rate_hz", ""):
            frame_rate = data_proc["parameters"]["movie_frame_rate_hz"]
        else:
            raise ValueError("Frame rate not found in processing metadata")
    return frame_rate


if __name__ == "__main__":
    start_time = dt.now(tz.utc)
    # Set the log level and name the logger
    logger = logging.getLogger(
        "Source extraction using Suite2p with or without Cellpose"
    )
    logger.setLevel(logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=str, help="Input directory", default="../data/"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="../results/"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="/scratch",
        help="Directory into which to write temporary files "
        "produced by Suite2P (default: /scratch)",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=0,
        help="Diameter that will be used for cellpose. "
        "If set to zero, diameter is estimated.",
    )
    parser.add_argument(
        "--anatomical_only",
        type=int,
        default=2,
        help="If greater than 0, specifies what to use Cellpose on. "
        "1: Will find masks on max projection image divided by mean image "
        "2: Will find masks on mean image "
        "3: Will find masks on enhanced mean image "
        "4: Will find masks on maximum projection image",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Whether or not binned movie should be denoised before cell detection.",
    )
    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=0.0,
        help="Threshold for cell detection that will be used by cellpose.",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=1.5,
        help="Flow threshold that will be used by cellpose.",
    )
    parser.add_argument(
        "--spatial_hp_cp",
        type=int,
        default=0,
        help="Window for spatial high-pass filtering of image "
        "to be used for cellpose.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="cyto",
        help="Path to pretrained model or string for model type "
        "(can be user’s model).",
    )
    parser.add_argument(
        "--use_suite2p_neuropil",
        action="store_true",
        help="Whether to use the fix weight provided by suite2p for neuropil \
        correction. If not, we use a mutual information based method.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    session, data_description, processing = get_metdata(input_dir)
    if len(list(input_dir.glob("*/decrosstalk/*decrosstalk.h5"))):
        input_fn = next(input_dir.glob("*/decrosstalk/*decrosstalk.h5"))
    else:
        input_fn = next(input_dir.glob("*/motion_correction/*registered.h5"))
    parent_directory = input_fn.parent
    if session is not None and "Bergamo" in session["rig_id"]:
        motion_corrected_fn = bergamo_segmentation(input_fn, session, temp_dir=tmp_dir)
    else:
        motion_corrected_fn = input_fn
    if data_description is not None:
        unique_id = "_".join(str(data_description["name"]).split("_")[-3:])
    else:
        unique_id = parent_directory.parent.name
    frame_rate = get_frame_rate(processing)

    output_dir = make_output_directory(output_dir, unique_id)
    # Set suite2p args.
    suite2p_args = suite2p.default_ops()
    # Overwrite the parameters for suite2p that are exposed
    suite2p_args["diameter"] = args.diameter
    suite2p_args["anatomical_only"] = args.anatomical_only
    suite2p_args["cellprob_threshold"] = args.cellprob_threshold
    suite2p_args["flow_threshold"] = args.flow_threshold
    suite2p_args["spatial_hp_cp"] = args.spatial_hp_cp
    suite2p_args["pretrained_model"] = args.pretrained_model
    suite2p_args["denoise"] = args.denoise
    suite2p_args["save_path0"] = args.tmp_dir
    # Here we overwrite the parameters for suite2p that will not change in our
    # processing pipeline. These are parameters that are not exposed to
    # minimize code length. Those are not set to default.
    suite2p_args["h5py"] = str(motion_corrected_fn)
    suite2p_args["data_path"] = []
    suite2p_args["roidetect"] = True
    suite2p_args["do_registration"] = 0
    suite2p_args["spikedetect"] = False
    suite2p_args["fs"] = frame_rate
    suite2p_args["neuropil_extract"] = True

    # determine nbinned from bin_duration and fs
    suite2p_args["bin_duration"] = 3.7  # The duration of time (in seconds) that
    # should be considered 1 bin for Suite2P ROI detection purposes. Requires
    # a valid value for 'fs' in order to derive an
    # nbinned Suite2P value. This allows consistent temporal downsampling
    # across movies with different lengths and/or frame rates.
    with h5py.File(suite2p_args["h5py"], "r") as f:
        nframes = f["data"].shape[0]
    bin_size = suite2p_args["bin_duration"] * suite2p_args["fs"]
    suite2p_args["nbinned"] = int(nframes / bin_size)
    logger.info(
        f"Movie has {nframes} frames collected at "
        f"{suite2p_args['fs']} Hz. "
        "To get a bin duration of "
        f"{suite2p_args['bin_duration']} "
        f"seconds, setting nbinned to "
        f"{suite2p_args['nbinned']}."
    )

    logger.info(f"running Suite2P v{suite2p.version}")
    try:
        suite2p.run_s2p(suite2p_args)
    except IndexError:  # raised when no ROIs found
        pass

    # load in the rois from the stat file and movie path for shape
    with h5py.File(str(motion_corrected_fn), "r") as open_vid:
        dims = open_vid["data"][0].shape
    if len(list(Path(args.tmp_dir).rglob("stat.npy"))):
        suite2p_stat_path = str(next(Path(args.tmp_dir).rglob("stat.npy")))
        suite2p_stats = np.load(suite2p_stat_path, allow_pickle=True)
        if session is not None and "Bergamo" in session["rig_id"]:
            # extract signals for all frames, not just those used for cell detection
            stat, traces_roi, traces_neuropil, _, _ = (
                suite2p.extraction.extraction_wrapper(
                    suite2p_stats, h5py.File(input_fn)["data"], ops=suite2p_args
                )
            )
        else:  # all frames have already been used for detection as well as extraction
            suite2p_f_path = str(next(Path(args.tmp_dir).rglob("F.npy")))
            suite2p_fneu_path = str(next(Path(args.tmp_dir).rglob("Fneu.npy")))
            traces_roi = np.load(suite2p_f_path, allow_pickle=True)
            traces_neuropil = np.load(suite2p_fneu_path, allow_pickle=True)
        iscell = np.load(str(next(Path(args.tmp_dir).rglob("iscell.npy"))))
        if args.use_suite2p_neuropil:
            traces_corrected = traces_roi - suite2p_args["neucoeff"] * traces_neuropil
            r_values = suite2p_args["neucoeff"] * np.ones(traces_roi.shape[0])
        else:
            traces_corrected, r_values, raw_r = get_FC_from_r(
                traces_roi, traces_neuropil
            )
        # convert ROIs to sparse COO 3D-tensor a la https://sparse.pydata.org/en/stable/construct.html
        data = []
        coords = []
        neuropil_coords = []
        for i, roi in enumerate(suite2p_stats):
            data.append(roi["lam"])
            coords.append(
                np.array(
                    [i * np.ones(len(roi["lam"])), roi["ypix"], roi["xpix"]],
                    dtype=np.int16,
                )
            )
            neuropil_coords.append(
                np.array(
                    [
                        i * np.ones(len(roi["neuropil_mask"])),
                        roi["neuropil_mask"] // dims[1],
                        roi["neuropil_mask"] % dims[1],
                    ],
                    dtype=np.int16,
                )
            )
        keys = list(suite2p_stats[0].keys())
        for k in ("ypix", "xpix", "lam", "neuropil_mask"):
            keys.remove(k)
        stat = {}
        for k in keys:
            stat[k] = [s[k] for s in suite2p_stats]
        data = np.concatenate(data)
        coords = np.hstack(coords)
        neuropil_coords = np.hstack(neuropil_coords)
        stat["soma_crop"] = np.concatenate(stat["soma_crop"])
        stat["overlap"] = np.concatenate(stat["overlap"])
    else:  # no ROIs found
        traces_roi, traces_neuropil, traces_corrected = [
            np.empty((0, nframes), dtype=np.float32)
        ] * 3
        r_values, data, coords, neuropil_coords, iscell = [[]] * 5
        if not args.use_suite2p_neuropil:
            raw_r = []
        keys = []

    cellpose_path = str(next(Path(args.tmp_dir).glob("**/cellpose.npz")))
    # write output files
    with (
        h5py.File(output_dir / "extraction.h5", "w") as f,
        np.load(cellpose_path) as cp,
    ):
        # traces
        f.create_dataset("traces/corrected", data=traces_corrected, compression="gzip")
        f.create_dataset("traces/neuropil", data=traces_neuropil, compression="gzip")
        f.create_dataset("traces/roi", data=traces_roi, compression="gzip")
        f.create_dataset("traces/neuropil_rcoef", data=r_values)
        if not args.use_suite2p_neuropil:
            # We save the raw r values if we are not using the suite2p neuropil.
            # This is useful for debugging purposes.
            f.create_dataset("traces/raw_neuropil_rcoef_mutualinfo", data=raw_r)
        for k in keys:
            dtype = np.array(stat[k]).dtype
            if dtype != "bool":
                dtype = "i2" if np.issubdtype(dtype, np.integer) else "f4"
            if k in ("skew", "std"):
                f.create_dataset(f"traces/{k}", data=stat[k], dtype=dtype)
            # ROIs
            else:
                f.create_dataset(f"rois/{k}", data=stat[k], dtype=dtype)
        f.create_dataset("rois/coords", data=coords, compression="gzip")
        f.create_dataset("rois/data", data=data, compression="gzip")
        f.create_dataset(
            "rois/shape", data=np.array([len(traces_roi), *dims], dtype=np.int16)
        )  # neurons x height x width
        f.create_dataset(
            "rois/neuropil_coords", data=neuropil_coords, compression="gzip"
        )
        # cellpose
        for k in cp.keys():
            f.create_dataset(f"cellpose/{k}", data=cp[k], compression="gzip")
        # classifier
        f.create_dataset(f"iscell", data=iscell, dtype="f4")

    write_output_metadata(
        vars(args),
        str(parent_directory),
        ProcessName.VIDEO_ROI_TIMESERIES_EXTRACTION,
        input_fn,
        output_dir / "extraction.h5",
        start_time,
    )
