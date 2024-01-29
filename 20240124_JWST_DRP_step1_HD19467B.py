import os
import time
import jwst
from jwst.pipeline import Detector1Pipeline
from glob import glob

if __name__ == "__main__":
    # Print out what pipeline version we're using
    print('JWST pipeline version',jwst.__version__)
    # targetdir = "/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/"
    # targetdir = "/stow/jruffio/data/JWST/nirspec/HD_19467/HD18511_post/"
    targetdir = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/"

    det1_dir = os.path.join(targetdir,"20240124_stage1") # Detector1 pipeline outputs will go here

    # We need to check that the desired output directories exist, and if not create them
    if not os.path.exists(det1_dir):
        os.makedirs(det1_dir)

    uncal_files = glob(os.path.join(targetdir,"uncal","jw0*_uncal.fits"))
    uncal_files.sort()
    for uncal_file in uncal_files:
        print(uncal_file)
    print('Found ' + str(len(uncal_files)) + ' input files to process')

    # Start a timer to keep track of runtime
    time0 = time.perf_counter()
    print(time0)

    for file in uncal_files[0::]:
        print(file)

        det1 = Detector1Pipeline() # Instantiate the pipeline

        #output directory
        det1.output_dir = det1_dir # Specify where the output should go

        #defining used pipeline steps
        det1.group_scale.skip = False  #scaling factor from group images
        det1.dq_init.skip = False      #data quality check
        det1.saturation.skip = False   #check for saturated pixels
        det1.ipc.skip = False          #interpixel capacitance scaling from a reference.
        det1.superbias.skip = False    #subtract a bias frame from the data
        det1.refpix.skip = False       #correct for drifts in read outs using reference pixels or columns
        det1.linearity.skip = False    #
        det1.persistence.skip = False  #
        det1.dark_current.skip = False #
        det1.jump.skip = False         #
        det1.ramp_fit.skip = False     #
        det1.gain_scale.skip = False   #


        #skipped steps that are for MIRI
        det1.firstframe.skip = True    #
        det1.lastframe.skip = True     #
        det1.reset.skip = True         #
        det1.rscd.skip = True          # Performs an RSCD correction to MIRI data. Skipped

        #choose what results to save and from what steps
        #det1.dq_init.save_results = True # save the data quality flag information
        det1.save_results = True # Save the final resulting _rate.fits files

        # det1.group_scale.save_results = True  #scaling factor from group images
        # det1.dq_init.save_results = True      #data quality check
        # det1.saturation.save_results = True   #check for saturated pixels
        # det1.ipc.save_results = True          #interpixel capacitance scaling from a reference.
        # det1.superbias.save_results = True    #subtract a bias frame from the data
        # det1.refpix.save_results = True       #correct for drifts in read outs using reference pixels or columns
        # det1.linearity.save_results = True    #
        # det1.persistence.save_results = True  #
        # det1.dark_current.save_results = True #
        # det1.jump.save_results = True         #
        # det1.ramp_fit.save_results = True     #
        # det1.gain_scale.save_results = True   #

        det1.saturation.n_pix_grow_sat = 0
        # det1.jump.maximum_cores = 8

        det1.run(file)
        # exit()

    # Print out the time benchmark
    time1 = time.perf_counter()
    print(f"Runtime so far: {time1 - time0:0.4f} seconds")
