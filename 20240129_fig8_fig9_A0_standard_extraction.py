import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
from glob import glob
import multiprocessing as mp
import astropy.units as u
from astropy import constants as const
import scipy
from scipy.stats import median_abs_deviation

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_cal import fitpsf

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar


if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    ####################
    ## To be modified
    ####################
    # Number of threads to be used for multithreading
    numthreads = 20
    # Number of nodes
    nodes = 40
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # utility folder where the intermediate and final data product will be saved
    utils_dir = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/breads/20240127_utils_clean/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    # Suffix to be added to the output filename that contained the extracted spectrum and centroid
    filename_suffix = "_webbpsf"
    # List of stage 2 cal.fits files to be extracted
    filelist = glob("/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/20240124_stage2/jw01128009001_03108_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    # Spectrum obtained privately from Brittany Miles for the same dataset, set to None if you don't have it
    ers_spectrum = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/20231009_Brittany_ERS_Spectra/A0_dithers_g140h-f100lp-g235h-f170lp-g395h-f290lp_x1d.fits"
    # ers_spectrum = None # skip it
    # Reference spectrum for A0 standard star TYC 4433-1800-1
    # Download from https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/
    stis_spectrum = "/stow/jruffio/models/TYC_4433-1800-1/1808347_stiswfc_004.fits"
    ####################
    ## No need to change for TYC_4433-1800-1 prog. id 1128
    # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
    wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
    wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
    # offset of the coordinates because point cloud is not centered
    init_centroid_nrs1 = [-0.25973700664819993, 0.7535417070247359]
    init_centroid_nrs2 = [-0.2679950725373308, 0.7554649479920329]
    # Flux Calibration parameters
    flux_calib_paras = -0.03745375, 1.08816133
    ####################

    # Define a multiprocessing pool for multi-threading when needed below
    mypool = mp.Pool(processes=numthreads)

    #################################
    # Spectral Extraction: Section of the code to run the webbPSF fitting
    if 1:
        # Looping over both NIRSpec detectors
        for detector in ["nrs1","nrs2"]:
            if detector == "nrs1":
                init_centroid = init_centroid_nrs1
                wv_sampling = wv_sampling_nrs1
            elif detector == "nrs2":
                init_centroid = init_centroid_nrs2
                wv_sampling = wv_sampling_nrs2

            # List of preprocessing steps to be applied to each cal.fits file
            # It's possible to call each step directly as dataobj.compute_xyz(parameters,..) if you prefer.
            # This is only for convenience.
            preproc_task_list = []
            preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
            preproc_task_list.append(["compute_coordinates_arrays"])
            preproc_task_list.append(["convert_MJy_per_sr_to_MJy"]) # old reduction, already in MJy
            # preproc_task_list.append(["apply_coords_offset",{"coords_offset":(ra_corr,dec_corr)}]) #-0.11584366936455087, 0.07189009712128012
            preproc_task_list.append(["compute_webbpsf_model",
                                      {"wv_sampling": wv_sampling, "image_mask": None, "pixelscale": 0.1, "oversample": 10,
                                       "parallelize": False, "mppool": mypool}, True, True])
            # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
            # preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
            preproc_task_list.append(["compute_starspectrum_contnorm", {"mppool": mypool}, True, True])
            preproc_task_list.append(["compute_starsubtraction",{"mppool":mypool},True,True])
            preproc_task_list.append(["compute_interpdata_regwvs",{"wv_sampling":wv_sampling},True,True])

            # Select only the files corresponding the correct detector
            nrs_filelist = []
            for filename in filelist:
                if detector in filename:
                    nrs_filelist.append(filename)
            nrs_filelist.sort()

            # Definie the filename of the output file saved by fitpsf
            splitbasename = os.path.basename(nrs_filelist[0]).split("_")
            fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")

            # Make a list of all the data objects to be given to fit psf
            dataobj_list = []
            for filename in nrs_filelist:
                print(filename)
                dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                          save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
                dataobj_list.append(dataobj)

            #Loading webbpsf, which is what will be fitted to the combined point cloud of dataobj_list
            with fits.open(dataobj.default_filenames["compute_webbpsf_model"]) as hdulist:
                wpsfs = hdulist[0].data
                wpsfs_header = hdulist[0].header
                wepsfs = hdulist[1].data
                webbpsf_wvs = hdulist[2].data
                webbpsf_X = hdulist[3].data
                webbpsf_Y = hdulist[4].data
                webbpsf_X = np.tile(webbpsf_X[None,:,:],(wepsfs.shape[0],1,1))
                webbpsf_Y = np.tile(webbpsf_Y[None,:,:],(wepsfs.shape[0],1,1))

            # Fit a model PSF (WebbPSF) to the combined point cloud of dataobj_list
            # Save output as fitpsf_filename
            ann_width = None
            padding = 0.0
            sector_area = None
            fitpsf(dataobj_list,wepsfs,webbpsf_X,webbpsf_Y, out_filename=fitpsf_filename,load=False,IWA = 0.0,OWA = 0.5,
                   mppool=mypool,init_centroid=init_centroid,run_init=False,ann_width=ann_width,padding=padding,
                   sector_area=sector_area,RDI_folder_suffix=filename_suffix,rotate_psf=dataobj_list[0].east2V2_deg,
                   flipx=True,psf_spaxel_area=dataobj_list[0].webbpsf_spaxel_area)

    #################################
    # Make plots
    if 1:
        hdulist = fits.open(stis_spectrum)
        from astropy.table import Table
        stis_table = Table(fits.getdata(stis_spectrum,1))
        stis_wvs =  (np.array(stis_table["WAVELENGTH"]) *u.Angstrom).to(u.um).value # angstroms -> mum
        stis_spec = np.array(stis_table["FLUX"]) * u.erg /u.s/u.cm**2/u.Angstrom # erg s-1 cm-2 A-1
        stis_spec = stis_spec.to(u.W*u.m**-2/u.um)
        stis_spec_Fnu = stis_spec*(stis_wvs*u.um)**2/const.c # from Flambda back to Fnu
        stis_spec_Fnu = stis_spec_Fnu.to(u.MJy).value

        splitbasename = os.path.basename(filelist[0]).split("_")
        fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")
        if "nrs1" in fitpsf_filename:
            fitpsf_filename_nrs1 = fitpsf_filename
            fitpsf_filename_nrs2 = fitpsf_filename.replace("nrs1","nrs2")
        elif "nrs2" in fitpsf_filename:
            fitpsf_filename_nrs1 = fitpsf_filename.replace("nrs2","nrs1")
            fitpsf_filename_nrs2 = fitpsf_filename

        flux_calib_wvs = []
        flux_calib = []
        flux2save_wvs = []
        flux2save = []

        for det_id,(_detector,_fitpsf_filename,wv_sampling) in enumerate(zip(["nrs1","nrs2"],[fitpsf_filename_nrs1,fitpsf_filename_nrs2],[wv_sampling_nrs1,wv_sampling_nrs2])):
            # /stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/breads/20240127_utils_clean/jw01128009001_03108_nrs1_fitpsf_webbpsf.fits
            # /stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/breads/20240127_utils_clean/jw01128009001_03104_nrs1_fitpsf_webbpsf.fits
            with fits.open(_fitpsf_filename) as hdulist:
                bestfit_coords = hdulist[0].data
                wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
                wpsf_ra_offset = hdulist[0].header["INIT_RA"]
                wpsf_dec_offset = hdulist[0].header["INIT_DEC"]
                all_interp_psfsub = hdulist[1].data
                all_interp_psfmodel = hdulist[2].data

            color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
            print(bestfit_coords.shape)
            fontsize=12
            plt.figure(1,figsize=(12,10))
            gs = gridspec.GridSpec(7,1, height_ratios=[1,1,0.5,0.3,1,1,0.5], width_ratios=[1])
            gs.update(left=0.075, right=0.95, bottom=0.07, top=0.95, wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[3*det_id+det_id, 0])

            flux2save_wvs.extend(wv_sampling)
            flux2save.extend(bestfit_coords[0,:,0])

            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9,linestyle="-",color=color_list[0],label="Fixed centroid",linewidth=1)
            plt.plot(wv_sampling,bestfit_coords[0,:,1]*1e9,linestyle="--",color=color_list[2],label="Free centroid",linewidth=1)
            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9*np.polyval(flux_calib_paras, wv_sampling),linestyle="-.",color=color_list[1],label="Fixed centroid - Corrected",linewidth=2)
            plt.plot(stis_wvs,stis_spec_Fnu*1e9,linestyle=":",color="black",label="CALSPEC",linewidth=2)
            plt.xlim([wv_sampling[0],wv_sampling[-1]])
            # plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("Flux density (mJy)",fontsize=fontsize)
            # plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            if det_id == 0:
                plt.ylim(4.5,10.5)
                plt.legend(loc="upper right")
                plt.text(0.01,0.01, '4 dithers - TYC 4433-1800-1', fontsize=fontsize, ha='left', va='bottom',color="black", transform=plt.gca().transAxes)# \n 36-166 $\mu$Jy
            if det_id == 1:
                plt.ylim(3.0,5.5)
            plt.xticks([])

            ax1 = plt.subplot(gs[3*det_id+det_id+1, 0])
            if ers_spectrum is not None:
                data = fits.getdata(ers_spectrum)
                ERS_wavelengths = data['wavelength']
                ERS_fluxes_MJy  = data['flux']/1e6
                error  = data['FLUX_ERROR']/1e6

                ERS_fluxes_MJy_smooth = scipy.ndimage.median_filter(ERS_fluxes_MJy, size=10, footprint=None, output=None, mode='reflect')
                meddev = median_abs_deviation((ERS_fluxes_MJy-ERS_fluxes_MJy_smooth))
                threshold=10
                where_bad = np.where((np.abs((ERS_fluxes_MJy-ERS_fluxes_MJy_smooth)) / meddev > threshold)|(ERS_wavelengths<2.88)|((ERS_wavelengths>4.08)*(ERS_wavelengths<4.12)))
                ERS_fluxes_MJy[where_bad] = np.nan
                plt.plot(ERS_wavelengths, ERS_fluxes_MJy*1e9,color=color_list[0],linewidth=2, label="Updated Miles+2023",dashes=[5, 3, 10, 3])
            plt.plot(stis_wvs,stis_spec_Fnu*1e9,linestyle=":",color="black",label="CALSPEC",linewidth=2)
            plt.xlim([wv_sampling[0],wv_sampling[-1]])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            # plt.ylabel("Flux density (mJy)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            if det_id == 0:
                plt.ylim(4.5,10.5)
                plt.legend(loc="upper right")
            if det_id == 1:
                # plt.legend(loc="upper right")
                plt.ylim(3.0,5.5)
                plt.yticks([3.0,3.5,4.0,4.5,5.0])

            ax1 = plt.subplot(gs[3*det_id+det_id+2, 0])
            calspec_func = interp1d(stis_wvs,stis_spec_Fnu*1e9)
            plt.plot(ERS_wavelengths, ERS_fluxes_MJy*1e9-calspec_func(ERS_wavelengths),color=color_list[0],linewidth=1, label="Updated Miles+2023",dashes=[5, 3, 10, 3])#,linestyle="-."
            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9*np.polyval(flux_calib_paras, wv_sampling)-calspec_func(wv_sampling),linestyle="-.",color=color_list[1],label="Fixed centroid - Corrected",linewidth=1)
            plt.xlim([wv_sampling[0],wv_sampling[-1]])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("Difference (mJy)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.ylim(-0.5,0.5)
            if det_id == 0:
                plt.yticks((-0.5,0.0, 0.5))
                plt.legend(loc="upper right")
            if det_id == 1:
                plt.yticks((-0.5,0.0))
            #     # plt.legend(loc="upper right")
            #     plt.ylim(3.0,5.5)
            #     plt.yticks([3.0,3.5,4.0,4.5,5.0])

            plt.figure(2,figsize=(6,6))
            plt.subplot(2,1,1)
            plt.plot(wv_sampling,bestfit_coords[0,:,2],label=_detector)
            # plt.plot(wv_sampling,bestfit_coords[1,:,2])
            # plt.plot(wv_sampling,bestfit_coords[:,2])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("$\Delta$RA (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.ylim([-0.280,-0.250])
            plt.legend(loc="upper right")
            plt.xlim([2.83,5.28])
            if det_id == 1:
                plt.text(0.01,0.01, '4 dithers - TYC 4433-1800-1', fontsize=fontsize, ha='left', va='bottom',color="black", transform=plt.gca().transAxes)# \n 36-166 $\mu$Jy
                plt.legend(loc="upper right")
            print("centroid",np.nanmedian(bestfit_coords[0,:,2]),np.nanmedian(bestfit_coords[0,:,3]))

            plt.subplot(2,1,2)
            plt.plot(wv_sampling,bestfit_coords[0,:,3])
            # plt.plot(wv_sampling,bestfit_coords[1,:,3])
            # plt.plot(wv_sampling,bestfit_coords[:,3])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.ylim([0.745,0.760])
            plt.xlim([2.83,5.28])

            plt.figure(3)
            plt.subplot(2,2,1)
            plt.title("Ratios fixed/free centroid")
            plt.plot(wv_sampling,bestfit_coords[0,:,1]/bestfit_coords[0,:,0])
            plt.subplot(2,2,2)
            plt.title("Ratios with Calspec")
            interp_stis_spec = interp1d(stis_wvs,stis_spec_Fnu,bounds_error=False,fill_value=np.nan)(wv_sampling)
            plt.plot(wv_sampling,bestfit_coords[0,:,0]/interp_stis_spec)
            plt.plot(wv_sampling,bestfit_coords[0,:,1]/interp_stis_spec,linestyle="--")
            flux_calib_wvs.extend(wv_sampling)
            flux_calib.extend(interp_stis_spec / bestfit_coords[0, :, 0])
            plt.subplot(2,2,3)
            plt.title("displacement")
            plt.plot(wv_sampling,np.sqrt((bestfit_coords[0,:,2]-(-0.256))**2+(bestfit_coords[0,:,3]-(0.751))**2))
            plt.subplot(2,2,4)
            plt.title("diff with Calspec")
            plt.plot(wv_sampling,interp_stis_spec -bestfit_coords[0,:,0])
            # plt.plot(wv_sampling,bestfit_coords[0,:,0] - interp_stis_spec)
            plt.plot(wv_sampling,interp_stis_spec - bestfit_coords[0,:,1],linestyle="--")
            # plt.plot(wv_sampling,bestfit_coords[0,:,1] - interp_stis_spec,linestyle="--")

        flux_calib_wvs,flux_calib = np.array(flux_calib_wvs),np.array(flux_calib)
        wherefinite = np.where(np.isfinite(flux_calib))
        flux_calib_wvs, flux_calib=flux_calib_wvs[wherefinite],flux_calib[wherefinite]
        # Deriving the parameters of the flux calibration
        print("flux calibration", np.polyfit(flux_calib_wvs,flux_calib ,deg=1))
        # flux calibration [-0.03745375  1.08816133]


        plt.figure(1)
        plt.tight_layout()
        plt.figure(2)
        plt.tight_layout()
        # plt.show()

        out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
        plt.figure(1)
        out_filename = os.path.join(out_png, "photocalib.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

        plt.figure(2)
        # plt.tight_layout()
        out_filename = os.path.join(out_png, "centroid.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

        # out_filename = os.path.join(out_png, "TYC_4433-1800-1_spectrum_microns_MJy.png")
        # hdulist = fits.HDUList()
        # hdulist.append(fits.PrimaryHDU(data=flux2save_wvs))
        # hdulist.append(fits.ImageHDU(data=flux2save))
        # try:
        #     hdulist.writeto(out_filename, overwrite=True)
        # except TypeError:
        #     hdulist.writeto(out_filename, clobber=True)
        # hdulist.close()
        #
        # plt.figure(10)
        # with fits.open(out_filename) as hdulist:
        #     wvs = hdulist[0].data
        #     spec = hdulist[1].data
        # plt.plot(wvs,spec)

        # plt.subplot(2,2,3)
        # plt.plot(wv_sampling,bestfit_coords[:,2])
        # plt.subplot(2,2,4)
        # plt.plot(wv_sampling,bestfit_coords[:,3])
        plt.show()

    exit()