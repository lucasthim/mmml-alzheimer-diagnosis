
import os

import pandas as pd
import numpy as np
import ants
# antspyx >=0.4 moved ANTsImage out of the top-level namespace; restore ants.ANTsImage
# so the type annotations below resolve regardless of antspyx version.
if not hasattr(ants, 'ANTsImage'):
    from ants.core.ants_image import ANTsImage as _ANTsImage
    ants.ANTsImage = _ANTsImage

# Original Colab template path (kept for reference). The file no longer exists locally;
# resolve_atlas_path() below picks a real template at call time.
ATLAS_PATH = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/atlas/atlas_t1.nii'

def resolve_atlas_path():
    '''
    Locate the registration template (fixed image), in priority order:

    1. $ATLAS_PATH environment variable, if it points at an existing file.
    2. Repo-local data/mri/atlas/atlas_t1.nii (the original pipeline's atlas), if present.
    3. ANTsPy's bundled MNI152 T1 template (turnkey fallback, ~/.antspy/mni.nii.gz).

    Note: the original 2021 runs used a specific atlas_t1.nii. The MNI152 fallback is a
    standard, reproducible substitute but is NOT bit-identical to that template, so
    registrations will differ slightly from the original study. Drop the original atlas
    at data/mri/atlas/atlas_t1.nii (or set $ATLAS_PATH) to reproduce exactly.
    '''
    for candidate in (os.environ.get('ATLAS_PATH'), 'data/mri/atlas/atlas_t1.nii'):
        if candidate and os.path.exists(candidate):
            return candidate
    return ants.get_ants_data('mni')

def register_image_with_atlas(moving:ants.ANTsImage = None, type_of_transform = 'Affine')-> ants.ANTsImage:
    
    '''
    Execute ANTs Registration.

    Parameters
    ----------

    image: Image file in numpy array format. If provided, function will use it instead of input_path

    type_of_transform: Type of Registration Transformation to be applied. Values tested: 'Affine', 'Similarity', 'Rigid'.
    
    
    Returns
    ----------
    
    Registered image in the ANTsImage format.
    '''

    fixed = ants.image_read(resolve_atlas_path())
    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform=type_of_transform ,grad_step=0.1)
    warpedimage = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
    return warpedimage