import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as fits
import pandas as pd
from glob import glob
import multiprocessing as mp


from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_cal import fitpsf

from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # to check if the script worked, open the newly created files for NRS2 in ds9:
    # e.g.: /stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils/jw01414004001_02101_00001_nrs2_cal_regwvs_psfsub_refPSF1.fits
    # The CO lines for the companion HD 19467 B should be clearly visible in the PSF subtracted "****_nrs2_cal_regwvs_psfsub_refPSF1.fits" files

    ####################
    ## To be modified
    ####################
    # Number of threads to be used for multithreading
    numthreads = 20
    # Number of nodes
    nodes = 40
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # Suffix to be added to the output filename
    filename_suffix = "_refPSF1"
    if 1: # Science data: List of stage 2 cal.fits files
        filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
        filelist.sort()
        for filename in filelist:
            print(filename)
        print("N files: {0}".format(len(filelist)))
        # utility folder where the intermediate and final data product will be saved
        utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils/"
        if not os.path.exists(utils_dir):
            os.makedirs(utils_dir)
    if 1: # Reference star: List of stage 2 cal.fits files
        filelist_ref1 = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD18511_post/20240124_stage2_clean/jw01414005001_02101_*_nrs*_cal.fits")
        filelist_ref1.sort()
        for filename in filelist_ref1:
            print(filename)
        print("N ref files: {0}".format(len(filelist_ref1)))
        # utility folder where the intermediate and final data product will be saved
        utils_ref1_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils_ref1/"
        if not os.path.exists(utils_ref1_dir):
            os.makedirs(utils_ref1_dir)
    ####################
    ## No need to change for HD19467B prog. id 1414
    # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
    wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
    wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
    # List of empirical offsets to correct the wcs coordinate system
    sc_centroid_nrs1 = [-0.13499443, -0.07202978]
    sc_centroid_nrs2 = [-0.12986898, -0.08441719]
    ref_centroid_nrs1 = [-0.08714674, -0.06863246]
    ref_centroid_nrs2 = [-0.08415047, -0.08143505]
    ####################

    # Define a multiprocessing pool for multi-threading when needed below
    mypool = mp.Pool(processes=numthreads)


    #################################
    # Spectral Extraction: Section of the code to run the webbPSF fitting
    # Looping over both NIRSpec detectors
    for detector in ["nrs1","nrs2"]:
        if detector == "nrs1":
            wv_sampling = wv_sampling_nrs1
            sc_centroid = sc_centroid_nrs1
            ref_centroid = ref_centroid_nrs1
        elif detector == "nrs2":
            wv_sampling = wv_sampling_nrs2
            sc_centroid = sc_centroid_nrs2
            ref_centroid = ref_centroid_nrs1


        ## First, PREPROCESS THE REFERENCE STAR to get its point cloud
        if 1:
            # Select only the files corresponding the correct detector
            nrs_filelist_ref1 = []
            for filename in filelist_ref1:
                if detector in filename:
                    nrs_filelist_ref1.append(filename)
            nrs_filelist_ref1.sort()

            # Make sure the webbpsf is computed, but skip for later, we don't technically need it other than for computing a initial centroid
            dataobj0 = JWSTNirspec_cal(nrs_filelist_ref1[0], crds_dir=crds_dir, utils_dir=utils_ref1_dir,save_utils=True, load_utils=True)
            webbpsf_reload = dataobj0.reload_webbpsf_model()
            if webbpsf_reload is None:
                dataobj0.compute_webbpsf_model(wv_sampling=wv_sampling,image_mask=None,pixelscale=0.1,oversample=10,parallelize=False, mppool= mypool)

            # List of preprocessing steps to be applied to each cal.fits file
            # It's possible to call each step directly as dataobj.compute_xyz(parameters,..) if you prefer.
            # This is only for convenience.
            preproc_task_list = []
            preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
            preproc_task_list.append(["compute_coordinates_arrays"])
            preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])  # old reduction, already in MJy
            preproc_task_list.append(["apply_coords_offset", {
                "coords_offset": (ref_centroid[0], ref_centroid[1])}])  # -0.11584366936455087, 0.07189009712128012
            # preproc_task_list.append(["compute_webbpsf_model",
            #                           {"wv_sampling": wv_sampling, "image_mask": None, "pixelscale": 0.1, "oversample": 10,
            #                            "parallelize": False, "mppool": mypool}, True, True])
            # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
            preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}, True, True])
            preproc_task_list.append(["compute_starspectrum_contnorm", {"mppool": mypool}, True, True])
            preproc_task_list.append(["compute_starsubtraction", {"mppool": mypool}, True, True])
            preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True])

            # REFERENCE STAR
            # Make a list of all the data objects to be given to fit psf
            dataobj_ref1_list = []
            # cen_ref1_list = []
            for filename in nrs_filelist_ref1:
                dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_ref1_dir,
                                          save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
                dataobj_ref1_list.append(dataobj)

                # webbPSFfit_centroid =dataobj.reload_new_coords_from_webbPSFfit(apply_offset=False)
                # if webbPSFfit_centroid is None:
                #     webbPSFfit_centroid = dataobj.compute_new_coords_from_webbPSFfit(IWA=0.2,OWA=1.0,save_utils=True,apply_offset=False)
                # cen_ref1_list.append(webbPSFfit_centroid)

            # exit()

            # initialize what will become the big stacks for the combined point cloud using the first exposure
            dataobj0 = dataobj_ref1_list[0]
            reload_interpdata_outputs = dataobj0.reload_interpdata_regwvs(load_filename=dataobj0.default_filenames["compute_interpdata_regwvs"])
            all_interp_ref1_ra, all_interp_ref1_dec, all_interp_ref1_wvs, all_interp_ref1_flux, all_interp_ref1_err, all_interp_ref1_badpix, all_interp_ref1_area2d = reload_interpdata_outputs

            if len(dataobj_ref1_list) > 1:
                for dataobj in dataobj_ref1_list[1::]:
                    reload_interpdata_outputs = dataobj.reload_interpdata_regwvs(load_filename=dataobj.default_filenames["compute_interpdata_regwvs"])
                    interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = reload_interpdata_outputs

                    # Stack up all the point clouds together
                    all_interp_ref1_ra = np.concatenate((all_interp_ref1_ra, interp_ra), axis=0)
                    all_interp_ref1_dec = np.concatenate((all_interp_ref1_dec, interp_dec), axis=0)
                    all_interp_ref1_wvs = np.concatenate((all_interp_ref1_wvs, interp_wvs), axis=0)
                    all_interp_ref1_flux = np.concatenate((all_interp_ref1_flux, interp_flux), axis=0)
                    all_interp_ref1_err = np.concatenate((all_interp_ref1_err, interp_err), axis=0)
                    all_interp_ref1_badpix = np.concatenate((all_interp_ref1_badpix, interp_badpix), axis=0)
                    all_interp_ref1_area2d = np.concatenate((all_interp_ref1_area2d, interp_area2d), axis=0)

            all_interp_ref1_flux[np.where(np.isnan(all_interp_ref1_badpix))] = np.nan

            psf_spaxel_area = np.nanmedian(all_interp_ref1_area2d)
            all_interp_ref1_flux = all_interp_ref1_flux / all_interp_ref1_area2d * psf_spaxel_area
            all_interp_ref1_err = all_interp_ref1_err / all_interp_ref1_area2d * psf_spaxel_area

            # apply coordinate offset to ra/dec arrays
            # ra_offset, dec_offset = np.nanmean(cen_ref1_list, axis=0)
            all_interp_ref1_ra -= ref_centroid[0]
            all_interp_ref1_dec -=  ref_centroid[1]
            # print("coucou",np.nanmean(cen_ref1_list, axis=0))

        # Select only the files corresponding the correct detector
        nrs_filelist = []
        for filename in filelist:
            if detector in filename:
                nrs_filelist.append(filename)
        nrs_filelist.sort()

        # Make sure the webbpsf is computed, but skip for later, we don't technically need it other than for computing a initial centroid
        dataobj = JWSTNirspec_cal(nrs_filelist[0], crds_dir=crds_dir, utils_dir=utils_dir,save_utils=True, load_utils=True)
        webbpsf_reload = dataobj.reload_webbpsf_model()
        if webbpsf_reload is None:
            dataobj.compute_webbpsf_model(wv_sampling=wv_sampling, image_mask=None, pixelscale=0.1, oversample=10,parallelize=False, mppool=mypool)

        # List of preprocessing steps to be applied to each cal.fits file
        # It's possible to call each step directly as dataobj.compute_xyz(parameters,..) if you prefer.
        # This is only for convenience.
        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"]) # old reduction, already in MJy
        preproc_task_list.append(["apply_coords_offset",{"coords_offset":(sc_centroid[0],sc_centroid[1])}]) #-0.11584366936455087, 0.07189009712128012
        # preproc_task_list.append(["compute_webbpsf_model",
        #                           {"wv_sampling": wv_sampling, "image_mask": None, "pixelscale": 0.1, "oversample": 10,
        #                            "parallelize": False, "mppool": mypool}, True, True])
        # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
        preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}, True, True])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"mppool": mypool}, True, True])
        preproc_task_list.append(["compute_starsubtraction",{"mppool":mypool},True,True])
        preproc_task_list.append(["compute_interpdata_regwvs",{"wv_sampling":wv_sampling},True,True])

        dataobj_list = []
        # cen_list = []
        for filename in nrs_filelist:
            print(filename)
            dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                      save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
            dataobj_list.append(dataobj)

        #     webbPSFfit_centroid =dataobj.reload_new_coords_from_webbPSFfit(apply_offset=False)
        #     if webbPSFfit_centroid is None:
        #         webbPSFfit_centroid = dataobj.compute_new_coords_from_webbPSFfit(IWA=0.2,OWA=1.0,save_utils=True,apply_offset=False)
        #     cen_list.append(webbPSFfit_centroid)
        # print("coucou 2 ",np.nanmean(cen_list, axis=0))

        splitbasename = os.path.basename(nrs_filelist[0]).split("_")
        fitpsf_filename = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+"_fitpsf"+filename_suffix+".fits")


        ann_width, padding, sector_area = 0.2,0.05,0.5
        IWA = 0.3
        OWA = 3.0
        fitpsf(dataobj_list,all_interp_ref1_flux.T,all_interp_ref1_ra.T,all_interp_ref1_dec.T, out_filename=fitpsf_filename,IWA = IWA,OWA = OWA,
               mppool=None,init_centroid=[0,0],ann_width=ann_width,padding=padding,
               sector_area=sector_area,RDI_folder_suffix=filename_suffix,rotate_psf=0.0,linear_interp=True,psf_spaxel_area=psf_spaxel_area)

