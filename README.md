# Extraction

This capsule uses a mix-and-match approach to combine Cellpose, Suite2p, and CaImAn for cell detection and signal extraction.

## Input

All parameters are passed to `extraction.py` using `python extraction.py [parameters]`.  
They are defined in `parse_args()` at the top of `extraction.py` using argparse.  
The most important one is `--input-dir` which should point to a directory that includes an hdf5 file with a 'data' 3D array of the _registered_ (or, in the case of multi-plane imaging, _decrosstalked_) movie. 

## Output

The main output is the __`extraction.h5`__ file.   
For QC, the figures `detected_ROIs.png` and `detected_ROIs_withIDs.png` display contours of the detected ROIs over a selection of summary images.  
If the capsule is run with the `--contour_video` argument, a 10x sped-up video is also created, showing ROI contours overlaid on the raw data, along with ROI activity and the residual.
<br>
The `extraction.h5` file contains three groups: `traces`, `rois`, and `cellpose`, with the latter only present if Cellpose is used for initialization (`--init`). The exact contents further depend on the used `--neuropil` method, with the __bolded__ fields below always present.

__`traces`:__
- __`corrected`__: Neuropil-corrected fluorescence traces
- __`roi`__: Neuropil-corrupted raw fluorescence traces of each ROI
- __`neuropil`__: Neuropil fluorescence traces
- `neuropil_rcoef`: Coefficients used for neuropil correction
- `raw_neuropil_rcoef_mutualinfo`: Raw coefficients for neuropil correction (minimizing mutual information between corrected and neuropil trace)
- `skew`: Skewness of neuropil-corrected fluorescence traces
- `std`: Standard deviation of neuropil-corrected fluorescence traces

__`rois`:__
- __`data`__: Values of the (weighted) ROIs
- __`coords`__: Index locations of every `data` value
- __`shape`__: Shape of the ROI array, # neurons x height of FOV (in pixels) x width of FOV (in pixels)
- `neuropil_coords`: Index locations of neuropil masks
- `aspect_ratio`, `compact`, `footprint`, `med`, `mrs`, `mrs0`, `npix`, `npix_norm`, `npix_norm_no_crop`, `npix_soma`, `overlap`, `radius`, `solidity`, `soma_crop`: Additional ROI stats, see suite2p documentation

`cellpose`:
- `masks`: Labeled image, where 0=no masks; 1,2,…=mask labels.
- `flows_in_hsv`: XY flow in HSV 0-255
- `flows`: XY flows at each pixel
- `cellprob`: Cell probability at each pixel
- `final_locations`: Final pixel locations after Euler integration
- `styles`: Style vector summarizing each image

Additional datasets at the root of `extraction.h5` are:
- __`iscell`__: Specifies whether a ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
- __`meanImg`__: The mean image of the movie over time
- __`maxImg`__: The maximum intensity projection of the movie over time
