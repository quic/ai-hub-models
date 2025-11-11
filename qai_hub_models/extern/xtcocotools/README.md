Files in this folder are copied from xtcocotools 1.14.3.

xtcocotools is no longer maintained, and is not compatible with Python 3.12+.
The manual build process for newer python versions requires pre-installed packages (eg numpy 1.x, which doesn't support python 3.13+)

Therefore, we have copied the parts of xtcocotools we depend on:
    * Loading the dataset via the COCO class
    * Keypoints-based eval

Both of these do not require the C API to function.

The LICENSE for the copied code can be found at https://github.com/jin-s13/xtcocoapi/blob/d74033ff1635e9002133b2380862bc2b728584d2/LICENSE
