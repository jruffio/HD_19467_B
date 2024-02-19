import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.io.fits as fits
from glob import glob
import multiprocessing as mp

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_cal import build_cube

import matplotlib
matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    ###########
    # This script will extract a "spectral cube" from a combined point cloud
    # It will fit a webbpsf at each spatial position and wavelength

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
        # utility folder where the intermediate and final data product will be saved
        utils_ref1_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils_ref1/"
        if not os.path.exists(utils_ref1_dir):
            os.makedirs(utils_ref1_dir)
    # Suffix that was added to the RDI output filename in the 20240201_RDI_reference_star.py script
    filename_suffix = "_refPSF1"
    # Output filenames:
    # cube_filename_RDIsubsci is the RDI subtracted dataset
    cube_filename_RDIsubsci_suffix = "_cube_RDI_20240202"
    # cube_filename_sci is the original science PSF (no PSF subtraction)
    # This is used to measure the amount of starlight at each position
    cube_filename_sci_suffix = "_cube_sci_20240202"
    # cube_filename_sci_suffix = None # Set to None to not process the unsubtracted science PSF
    # cube_filename_ref is the reference PSF (no PSF subtraction)
    cube_filename_ref_suffix = "_cube_ref1_20240202"
    # cube_filename_ref_suffix = None # Set to None to not process the unsubtracted reference PSF
    # Sampling of the wavelength directions:
    ra_vec = np.arange(-3,1.5,0.05)
    dec_vec = np.arange(-3,1.5,0.05)
    # ra_vec = np.arange(-1.5,-1.1,0.1)# Partial cube around companion for testing
    # dec_vec = np.arange(-1.1,-0.7,0.1)
    # min max index of the wv_sampling for partial reduction/debugging
    debug_init = None
    debug_end = None
    # debug_init = 1000 # Run these for a first attempt to make sure it works
    # debug_end = 1010
    # RA Dec offset of the companion HD19467B
    ra_offset = -1332.871/1000. # ra offset in as
    dec_offset = -875.528/1000. # dec offset in as
        # Offsets for HD 19467 B from https://www.whereistheplanet.com/
        # RA Offset = -1332.871 +/- 10.886 mas
        # Dec Offset = -875.528 +/- 12.360 mas
        # Separation = 1593.703 +/- 9.530 mas
        # PA = 236.712 +/- 0.483 deg
        # Reference: Brandt et al. 2021
    # Flux Calibration parameters
    flux_calib_paras = -0.03864459,  1.09360589
    ####################

    if 0:
        RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/20230705_HD19467b_RDI_1dspectrum.fits"
        hdulist_sc = fits.open(RDI_spec_filename)
        RDI_spec_wvs = hdulist_sc[0].data
        RDI_spec = hdulist_sc[1].data
        RDI_err = hdulist_sc[2].data
        plt.figure(2)
        plt.plot(RDI_spec_wvs,RDI_spec,label="submitted spectrum",color="blue")
        plt.fill_between(RDI_spec_wvs,-RDI_err,RDI_err,label="submitted Error",color="blue",alpha=0.5)
        # plt.plot(RDI_spec_wvs,RDI_spec_wvs*0,color="black")

        cube_filelist_RDIsubsci = glob(os.path.join(utils_dir,"*"+cube_filename_RDIsubsci_suffix+"*.fits"))
        for det_id, cube_filename_RDIsubsci in enumerate(cube_filelist_RDIsubsci):
            if "nrs1" in cube_filename_RDIsubsci:
                detector = "nrs1"
            elif "nrs2" in cube_filename_RDIsubsci:
                detector = "nrs2"

            hdulist = fits.open(cube_filename_RDIsubsci)
            flux_cube = hdulist[0].data
            fluxerr_cube = hdulist['FLUXERR_CUBE'].data
            ra_grid = hdulist['RA'].data
            dec_grid = hdulist['DEC'].data
            wv_sampling = hdulist['WAVE'].data

            flux_im = np.nanmean(flux_cube, axis=0)

            plt.figure(1)
            plt.subplot(1,np.size(cube_filelist_RDIsubsci),det_id+1)
            plt.title("PSF-subtracted image "+detector)
            plt.imshow(flux_im,origin="lower")
            plt.clim([0,5e-11])
            plt.figure(2)
            kmax,lmax = np.unravel_index(np.nanargmax(flux_im*(((ra_grid-(-1.5))**2+(dec_grid-(-0.9))**2)<0.4)),flux_im.shape)
            print(kmax,lmax,ra_grid[kmax,lmax],dec_grid[kmax,lmax])
            r2comp_grid = np.sqrt((ra_grid-ra_grid[kmax,lmax])**2+(dec_grid-dec_grid[kmax,lmax])**2)
            r2star_grid = np.sqrt((ra_grid)**2+(dec_grid)**2)
            whereannulus = np.where((r2comp_grid>0.4)*(r2comp_grid<0.5))
            speckle_std = np.nanstd(flux_cube[:,whereannulus[0],whereannulus[1]],axis=1)
            speckle_med = np.nanmedian(flux_cube[:,whereannulus[0],whereannulus[1]],axis=1)
            plt.plot(wv_sampling,(flux_cube[:,kmax,lmax]-speckle_med)*np.polyval(flux_calib_paras, wv_sampling),label="new spectrum",color="orange")
            # plt.plot(wv_sampling,flux_cube[:,whereannulus[0][0],whereannulus[1][0]]-speckle_med,color="grey",alpha=0.05,label="new speckles")
            # plt.plot(wv_sampling,flux_cube[:,whereannulus[0],whereannulus[1]]-speckle_med[:,None],color="grey",alpha=0.05)
            error = (speckle_std-speckle_med)*np.polyval(flux_calib_paras, wv_sampling)
            plt.fill_between(wv_sampling,-error,error,label="new Error",color="orange",alpha=0.5)
        plt.figure(2)
        # plt.plot(RDI_spec_wvs,RDI_spec_wvs*0,color="black")
        plt.title("HD19467B spectrum")
        plt.legend()
        plt.ylabel("MJy")
        plt.show()
        exit()

    # Define a multiprocessing pool for multi-threading when needed below
    mypool = mp.Pool(processes=numthreads)

    #################################

    # Looping over both NIRSpec detectors
    for detector in ["nrs1", "nrs2"]:
    # for detector in ["nrs1"]:
        ## First, PREPROCESS THE SCIENCE sequence to get its point cloud
        if 1:
            # Select only the files corresponding the correct detector
            nrs_filelist = []
            for filename in filelist:
                if detector in filename:
                    nrs_filelist.append(filename)
            nrs_filelist.sort()

            splitbasename = os.path.basename(nrs_filelist[0]).split("_")
            fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")
            cube_filename_RDIsubsci = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+cube_filename_RDIsubsci_suffix+".fits")

            # initialize what will become the big stacks for the combined point cloud using the first exposure
            dataobj0 = JWSTNirspec_cal(nrs_filelist[0], crds_dir=crds_dir, utils_dir=utils_dir,save_utils=True,load_utils=True)
            reload_interpdata_outputs = dataobj0.reload_interpdata_regwvs(load_filename=dataobj0.default_filenames["compute_interpdata_regwvs"])
            all_interp_ra, all_interp_dec, all_interp_wvs, all_interp_flux, all_interp_err, all_interp_badpix, all_interp_area2d = reload_interpdata_outputs

            interpdata_psfsub_filename = dataobj0.default_filenames["compute_interpdata_regwvs"].replace(".fits","_psfsub"+filename_suffix+".fits")
            reload_interpdata_outputs = dataobj0.reload_interpdata_regwvs(load_filename=interpdata_psfsub_filename)
            _,_,_, all_interp_flux_psfsub, _,_,_ = reload_interpdata_outputs

            if len(nrs_filelist) > 1:
                for filename in nrs_filelist[1::]:
                    dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,save_utils=True,load_utils=True)
                    reload_interpdata_outputs = dataobj.reload_interpdata_regwvs(load_filename=dataobj.default_filenames["compute_interpdata_regwvs"])
                    interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = reload_interpdata_outputs

                    interpdata_psfsub_filename = dataobj.default_filenames["compute_interpdata_regwvs"].replace(".fits","_psfsub"+filename_suffix+".fits")
                    reload_interpdata_outputs = dataobj.reload_interpdata_regwvs(load_filename=interpdata_psfsub_filename)
                    _,_,_, interp_flux_psfsub, _,_,_ = reload_interpdata_outputs

                    # Stack up all the point clouds together
                    all_interp_ra = np.concatenate((all_interp_ra, interp_ra), axis=0)
                    all_interp_dec = np.concatenate((all_interp_dec, interp_dec), axis=0)
                    all_interp_wvs = np.concatenate((all_interp_wvs, interp_wvs), axis=0)
                    all_interp_flux = np.concatenate((all_interp_flux, interp_flux), axis=0)
                    all_interp_flux_psfsub = np.concatenate((all_interp_flux_psfsub, interp_flux_psfsub), axis=0)
                    all_interp_err = np.concatenate((all_interp_err, interp_err), axis=0)
                    all_interp_badpix = np.concatenate((all_interp_badpix, interp_badpix), axis=0)
                    all_interp_area2d = np.concatenate((all_interp_area2d, interp_area2d), axis=0)

                N_dithers = all_interp_flux.shape[0] / interp_flux.shape[0]


                # Load the webbPSF model (or compute if it does not yet exist)
                webbpsf_reload = dataobj0.reload_webbpsf_model()
                if webbpsf_reload is None:
                    webbpsf_reload = dataobj0.compute_webbpsf_model(wv_sampling=dataobj0.wv_sampling,image_mask=None,pixelscale=0.1,oversample=10,parallelize=False, mppool= mypool)
                wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale = webbpsf_reload
                webbpsf_X = np.tile(webbpsf_X[None, :, :], (wepsfs.shape[0], 1, 1))
                webbpsf_Y = np.tile(webbpsf_Y[None, :, :], (wepsfs.shape[0], 1, 1))

            # flux_cube,fluxerr_cube,ra_grid, dec_grid = \
            #     build_cube(dataobj0.wv_sampling,dataobj0.east2V2_deg, # wavelength sampling
            #                all_interp_ra, all_interp_dec, all_interp_flux_psfsub, all_interp_err, all_interp_badpix,N_dithers, # combined point cloud
            #                wepsfs, webbpsf_X, webbpsf_Y, # webbPSF model for flux extraction
            #                ra_vec, dec_vec, # spatial sampling of final cube
            #                out_filename=cube_filename_RDIsubsci,linear_interp=True,mppool=mypool,
            #                debug_init=debug_init,debug_end=debug_end)  # min max wavelength indices for partial extraction


            if cube_filename_sci_suffix is not None:
                cube_filename_sci = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+cube_filename_sci_suffix+".fits")
                flux_cube,fluxerr_cube, ra_grid, dec_grid = \
                    build_cube(dataobj0.wv_sampling,dataobj0.east2V2_deg,
                               all_interp_ra, all_interp_dec, all_interp_flux, all_interp_err, all_interp_badpix,N_dithers,
                               wepsfs, webbpsf_X, webbpsf_Y,
                               ra_vec, dec_vec,
                               out_filename=cube_filename_sci,linear_interp=True,mppool=mypool,
                               debug_init=debug_init,debug_end=debug_end)  # mypool




            if cube_filename_ref_suffix is not None:
                ## First, PREPROCESS THE REFERENCE STAR to get its point cloud
                if 1:
                    # Select only the files corresponding the correct detector
                    nrs_filelist_ref1 = []
                    for filename in filelist_ref1:
                        if detector in filename:
                            nrs_filelist_ref1.append(filename)
                    nrs_filelist_ref1.sort()

                    splitbasename = os.path.basename(nrs_filelist_ref1[0]).split("_")
                    cube_filename_ref = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+cube_filename_ref_suffix+".fits")

                    # initialize what will become the big stacks for the combined point cloud using the first exposure
                    dataobj0 = JWSTNirspec_cal(nrs_filelist_ref1[0], crds_dir=crds_dir, utils_dir=utils_ref1_dir,
                                               save_utils=True, load_utils=True)
                    reload_interpdata_outputs = dataobj0.reload_interpdata_regwvs(
                        load_filename=dataobj0.default_filenames["compute_interpdata_regwvs"])
                    all_interp_ref1_ra, all_interp_ref1_dec, all_interp_ref1_wvs, all_interp_ref1_flux, all_interp_ref1_err, all_interp_ref1_badpix, all_interp_ref1_area2d = reload_interpdata_outputs

                    if len(nrs_filelist_ref1) > 1:
                        for filename in nrs_filelist_ref1[1::]:
                            dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_ref1_dir,
                                                      save_utils=True, load_utils=True)
                            reload_interpdata_outputs = dataobj.reload_interpdata_regwvs(
                                load_filename=dataobj.default_filenames["compute_interpdata_regwvs"])
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

                # Load the webbPSF model (or compute if it does not yet exist)
                webbpsf_reload = dataobj0.reload_webbpsf_model()
                if webbpsf_reload is None:
                    webbpsf_reload = dataobj0.compute_webbpsf_model(wv_sampling=dataobj0.wv_sampling,image_mask=None,pixelscale=0.1,oversample=10,parallelize=False, mppool= mypool)
                wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale = webbpsf_reload
                webbpsf_X = np.tile(webbpsf_X[None, :, :], (wepsfs.shape[0], 1, 1))
                webbpsf_Y = np.tile(webbpsf_Y[None, :, :], (wepsfs.shape[0], 1, 1))

                flux_cube,fluxerr_cube, ra_grid, dec_grid = \
                    build_cube(dataobj0.wv_sampling,dataobj0.east2V2_deg,
                               all_interp_ref1_ra, all_interp_ref1_dec,all_interp_ref1_flux, all_interp_ref1_err, all_interp_ref1_badpix,N_dithers,
                               wepsfs, webbpsf_X, webbpsf_Y,
                               ra_vec, dec_vec,
                               out_filename=cube_filename_ref,linear_interp=True,mppool=mypool,
                               debug_init=debug_init,debug_end=debug_end)  # mypool


    # plot extracted spectrum
    if 1:
        # RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/20230705_HD19467b_RDI_1dspectrum.fits"
        # hdulist_sc = fits.open(RDI_spec_filename)
        # RDI_spec_wvs = hdulist_sc[0].data
        # RDI_spec = hdulist_sc[1].data
        # RDI_err = hdulist_sc[2].data
        # plt.figure(2)
        # plt.plot(RDI_spec_wvs,RDI_spec,label="submitted spectrum",color="blue")
        # plt.fill_between(RDI_spec_wvs,-RDI_err,RDI_err,label="submitted Error",color="blue",alpha=0.5)

        cube_filelist_RDIsubsci = glob(os.path.join(utils_dir,"*"+cube_filename_RDIsubsci_suffix+"*.fits"))
        for det_id, cube_filename_RDIsubsci in enumerate(cube_filelist_RDIsubsci):
            if "nrs1" in cube_filename_RDIsubsci:
                detector = "nrs1"
            elif "nrs2" in cube_filename_RDIsubsci:
                detector = "nrs2"

            hdulist = fits.open(cube_filename_RDIsubsci)
            flux_cube = hdulist[0].data
            fluxerr_cube = hdulist['FLUXERR_CUBE'].data
            ra_grid = hdulist['RA'].data
            dec_grid = hdulist['DEC'].data
            wv_sampling = hdulist['WAVE'].data

            flux_im = np.nanmean(flux_cube, axis=0)

            plt.figure(1)
            plt.subplot(1,np.size(cube_filelist_RDIsubsci),det_id+1)
            plt.title("PSF-subtracted image "+detector)
            plt.imshow(flux_im,origin="lower")
            plt.clim([0,5e-11])
            plt.figure(2)
            kmax,lmax = np.unravel_index(np.nanargmax(flux_im*(((ra_grid-ra_offset)**2+(dec_grid-dec_offset)**2)<0.4)),flux_im.shape)
            print(kmax,lmax,ra_grid[kmax,lmax],dec_grid[kmax,lmax])
            r2comp_grid = np.sqrt((ra_grid-ra_grid[kmax,lmax])**2+(dec_grid-dec_grid[kmax,lmax])**2)
            r2star_grid = np.sqrt((ra_grid)**2+(dec_grid)**2)
            whereannulus = np.where((r2comp_grid>0.4)*(r2comp_grid<0.5))
            speckle_std = np.nanstd(flux_cube[:,whereannulus[0],whereannulus[1]],axis=1)
            speckle_med = np.nanmedian(flux_cube[:,whereannulus[0],whereannulus[1]],axis=1)
            plt.plot(wv_sampling,(flux_cube[:,kmax,lmax]-speckle_med)*np.polyval(flux_calib_paras, wv_sampling),label="new spectrum",color="orange")
            # plt.plot(wv_sampling,flux_cube[:,whereannulus[0][0],whereannulus[1][0]]-speckle_med,color="grey",alpha=0.05,label="new speckles")
            # plt.plot(wv_sampling,flux_cube[:,whereannulus[0],whereannulus[1]]-speckle_med[:,None],color="grey",alpha=0.05)
            error = (speckle_std-speckle_med)*np.polyval(flux_calib_paras, wv_sampling)
            plt.fill_between(wv_sampling,-error,error,label="new Error",color="orange",alpha=0.5)
        plt.figure(2)
        # plt.plot(RDI_spec_wvs,RDI_spec_wvs*0,color="black")
        plt.title("HD19467B spectrum")
        plt.legend()
        plt.ylabel("MJy")
        plt.show()
        exit()
