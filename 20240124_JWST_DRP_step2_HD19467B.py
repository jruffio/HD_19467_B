import os
import time
import jwst
from jwst.pipeline import Spec2Pipeline
from glob import glob

if __name__ == "__main__":
    # Print out what pipeline version we're using
    print('JWST pipeline version',jwst.__version__)

    # targetdir = "/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/"
    targetdir = "/stow/jruffio/data/JWST/nirspec/HD_19467/HD18511_post/"
    # targetdir = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/"
    # det1_dir = os.path.join(targetdir,"20240124_stage1") # Detector1 pipeline outputs will go here
    # spec2_dir = os.path.join(targetdir,"20240124_stage2") # Spec2 pipeline outputs will go here
    det1_dir = os.path.join(targetdir,"20240124_stage1_clean") # Detector1 pipeline outputs will go here
    spec2_dir = os.path.join(targetdir,"20240124_stage2_clean") # Spec2 pipeline outputs will go here

    # We need to check that the desired output directories exist, and if not create them
    if not os.path.exists(spec2_dir):
        os.makedirs(spec2_dir)

    rate_files = glob(os.path.join(det1_dir,"jw0*_rate.fits"))
    rate_files.sort()
    for rate_file in rate_files:
        print(rate_file)
    print('Found ' + str(len(rate_files)) + ' input files to process')

    # Start a timer to keep track of runtime
    time0 = time.perf_counter()
    print(time0)

    for fid,rate_file in enumerate(rate_files):
        print(fid,rate_file)

        # Start a timer to keep track of runtime
        time0 = time.perf_counter()
        # Setting up steps and running the Spec2 portion of the pipeline.

        spec2 = Spec2Pipeline()
        spec2.output_dir = spec2_dir
        # spec2.assign_wcs.skip = False
        # spec2.bkg_subtract.skip = False
        # spec2.imprint_subtract.skip = False
        # spec2.msa_flagging.skip = False
        # # spec2.srctype.source_type = 'POINT'
        # spec2.flat_field.skip = False
        # spec2.pathloss.skip = False
        # spec2.photom.skip = False
        spec2.cube_build.skip = True#False#True
        spec2.extract_1d.skip = True
        # spec3.cube_build.coord_system = 'skyalign'
        # spec2.cube_build.coord_system='ifualign'

        ##extra steps that only apply to other instruments modes.
        spec2.extract_2d.skip = True
        spec2.master_background_mos.skip = True
        spec2.barshadow.skip = True
        spec2.wavecorr.skip = True
        spec2.straylight.skip = True
        spec2.fringe.skip = True
        spec2.residual_fringe.skip = True
        spec2.wfss_contam.skip = True
        spec2.resample_spec.skip = True

        spec2.save_bsub = True

        #choose what results to save and from what steps
        spec2.save_results = True
        spec2.run(rate_file)

    # Print out the time benchmark
    time1 = time.perf_counter()
    print(f"Runtime so far: {time1 - time0:0.4f} seconds")
