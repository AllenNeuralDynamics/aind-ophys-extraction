# Extraction

This capsule combines cellpose and suite2p to perform cell detection and signal extraction.

## Input

All parameters are passed to extraction.py using `python extraction.py [parameters]`.
All parameters are defined in __main__ using argparse. The most important one is
'input-dir' which should point to a directory that includes an hdf5 file with a 'data' 3D array 
of the _registered_ (or, in the case of multi-plane imaging, _decrosstalked_) movie. 

## Output

The main output is the `extraction.h5` file.

It contains three groups: `traces`, `rois`, and `cellpose`.

__`traces`:__
- `corrected`: Neuropil-corrected fluorescence traces
- `roi`: Neuropil-corrupted raw fluorescence traces of each ROI
- `neuropil`: Neuropil fluorescence traces

__`rois`:__
- `data`: Values of the weighted rois
- `coords`: Index locations of every `data` value
- `shape`: Shape of the ROI array, # neurons x height of FOV (in pixels) x width of FOV (in pixels)
- `aspect_ratio`, `compact`, `footprint`, `med`, `mrs`, `mrs0`, `npix`, `npix_norm`, `npix_norm_no_crop`, `npix_soma`, `overlap`, `radius`, `solidity`, `soma_crop`: Additional ROI stats, see suite2p documentation

__`cellpose`:__
- `masks`: Labeled image, where 0=no masks; 1,2,…=mask labels.
- `flows_in_hsv`: XY flow in HSV 0-255
- `flows`: XY flows at each pixel
- `cellprob`: Cell probability
- `final_locations`: Final pixel locations after Euler integration
- `styles`: Style vector summarizing each image