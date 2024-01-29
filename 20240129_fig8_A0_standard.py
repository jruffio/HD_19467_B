import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as fits
import pandas as pd
from glob import glob
import multiprocessing as mp
import h5py
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const
from scipy.signal import correlate2d

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.grid_search import grid_search
from breads.fm.hc_splinefm_jwst_nirspec import hc_splinefm_jwst_nirspec

from copy import copy
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

    numthreads = 20

    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"

    # Choose to reduce either detector (NRS1 or NRS2)
    detector,photfilter_name,wv_sampling = "nrs1", "F360M", np.arange(2.859509, 4.1012874, 0.0006763935)
    # detector,photfilter_name,wv_sampling= "nrs2", "F460M",np.arange(4.081285,5.278689,0.0006656647)


    # Definition of the nodes for the spline
    if detector == "nrs1":
        dw1,dw2 = 0.02,0.04
        l0,l2 = 2.859509-dw1, 4.1012874
        l1 = (l0+l2)/2
        x1_nodes = np.arange(l0,l1, dw1)
        l1 = x1_nodes[-1]+dw2
        x2_nodes = np.arange(l1,l2, dw2)
        x_nodes = np.concatenate([x1_nodes,x2_nodes])
    elif detector == "nrs2":
        dw1,dw2 = 0.04,0.02
        l0,l2 = 4.081285,5.278689+dw2
        l1 = (l0+l2)/2
        x1_nodes = np.arange(l0,l1, dw1)
        l1 = x1_nodes[-1]+dw2
        x2_nodes = np.arange(l1,l2, dw2)
        x_nodes = np.concatenate([x1_nodes,x2_nodes])
    print(x_nodes)
    print(np.size(x_nodes))

    if 1:
        utils_dir = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/breads/20240127_utils/"
        filelist = glob("/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/20240124_stage2_clean/jw01128009001_03108_*_"+detector+"_cal.fits")
        if not os.path.exists(utils_dir):
            os.makedirs(utils_dir)

        if detector == "nrs1":
            coords_offset = [-0.25973700664819993, 0.7535417070247359]
        elif detector == "nrs2":
            coords_offset = [-0.2679950725373308, 0.7554649479920329]
        if 0:
            ers_spectrum= "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/dithers_g395h-f290lp_x1d.fits"
            data = fits.getdata(ers_spectrum)
            wavelength1 = data['wavelength']
            flux1  = data['flux']/1e6
            error1  = data['FLUX_ERROR']/1e6
            ers_spectrum= "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/dithers_g395h-f290lp_x1d (1).fits"
            data = fits.getdata(ers_spectrum)
            wavelength2 = data['wavelength']
            flux2  = data['flux']/1e6
            error2  = data['FLUX_ERROR']/1e6
            ERS_wavelengths, ERS_fluxes_MJy = np.concatenate((wavelength1,wavelength2)), np.concatenate((flux1,flux2))
        else:
            ers_spectrum= "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/20231009_Brittany_ERS_Spectra/A0_dithers_g140h-f100lp-g235h-f170lp-g395h-f290lp_x1d.fits"
            data = fits.getdata(ers_spectrum)
            ERS_wavelengths = data['wavelength']
            ERS_fluxes_MJy  = data['flux']/1e6
            error  = data['FLUX_ERROR']/1e6
        import scipy
        ERS_fluxes_MJy_smooth = scipy.ndimage.median_filter(ERS_fluxes_MJy, size=10, footprint=None, output=None, mode='reflect')
        from scipy.stats import median_abs_deviation
        meddev = median_abs_deviation((ERS_fluxes_MJy-ERS_fluxes_MJy_smooth))
        threshold=10
        where_bad = np.where((np.abs((ERS_fluxes_MJy-ERS_fluxes_MJy_smooth)) / meddev > threshold)|(ERS_wavelengths<2.88)|((ERS_wavelengths>4.08)*(ERS_wavelengths<4.12)))
        ERS_fluxes_MJy[where_bad] = np.nan

        # plt.plot(ERS_wavelengths,ERS_fluxes_MJy_smooth)
        # plt.plot(ERS_wavelengths,ERS_fluxes_MJy)
        # plt.ylim([0,1e-8])
        # plt.show()
        # plt.plot(wavelength1,flux1)
        # plt.plot(wavelength2,flux2)
        # plt.ylim([-1e-10,1e-8])
        # plt.show()

        stis_spectrum = "/stow/jruffio/models/TYC_4433-1800-1/1808347_stiswfc_004.fits"
        # vega_filename =  "/data/osiris_data/low_res/alpha_lyr_stis_010.fits"
        hdulist = fits.open(stis_spectrum)
        from astropy.table import Table
        stis_table = Table(fits.getdata(stis_spectrum,1))
        stis_wvs =  (np.array(stis_table["WAVELENGTH"]) *u.Angstrom).to(u.um).value # angstroms -> mum
        stis_spec = np.array(stis_table["FLUX"]) * u.erg /u.s/u.cm**2/u.Angstrom # erg s-1 cm-2 A-1
        stis_spec = stis_spec.to(u.W*u.m**-2/u.um)
        stis_spec_Fnu = stis_spec*(stis_wvs*u.um)**2/const.c # from Flambda back to Fnu
        stis_spec_Fnu = stis_spec_Fnu.to(u.MJy).value


    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))

    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}

    photfilter = os.path.join(external_dir,"JWST_NIRCam."+photfilter_name+".dat")
    filter_arr = np.loadtxt(photfilter)
    trans_wvs = filter_arr[:,0]/1e4
    trans = filter_arr[:,1]
    photfilter_f = interp1d(trans_wvs,trans,bounds_error=False,fill_value=0)
    photfilter_wv0 = np.nansum(trans_wvs*photfilter_f(trans_wvs))/np.nansum(photfilter_f(trans_wvs))
    bandpass = np.where(photfilter_f(trans_wvs)/np.nanmax(photfilter_f(trans_wvs))>0.01)
    photfilter_wvmin,photfilter_wvmax = trans_wvs[bandpass[0][0]],trans_wvs[bandpass[0][-1]]
    print(photfilter_wvmin,photfilter_wvmax)
    # exit()

    mypool = mp.Pool(processes=numthreads)

    preproc_task_list = []
    preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
    preproc_task_list.append(["compute_coordinates_arrays"])
    preproc_task_list.append(["convert_MJy_per_sr_to_MJy"]) # old reduction, already in MJy
    ra_corr,dec_corr = -0.1285876002234175, - 0.06997868615326872
    preproc_task_list.append(["apply_coords_offset",{"coords_offset":(ra_corr,dec_corr)}]) #-0.11584366936455087, 0.07189009712128012
    preproc_task_list.append(["compute_webbpsf_model",
                              {"wv_sampling": wv_sampling, "image_mask": None, "pixelscale": 0.1, "oversample": 10,
                               "parallelize": False, "mppool": mypool}, True, True])
    # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
    preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
    preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": x_nodes, "mppool": mypool}, True, True])
    preproc_task_list.append(["compute_starsubtraction",{"mppool":mypool},True,True])
    preproc_task_list.append(["compute_interpdata_regwvs",{"wv_sampling":wv_sampling},True,True])


    dataobj_list = []
    # if 1:
    #     filename = filelist[0]
    cen_list = []
    for filename in filelist:#[0:1]:
        print(filename)

        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)

        dataobj_list.append(dataobj)
        cen_list.append([dataobj.wpsf_ra_offset,dataobj.wpsf_dec_offset])

    splitbasename = os.path.basename(filelist[0]).split("_")
    RDI_folder_suffix = "_webbpsf"
    fitpsf_filename = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+"_fitpsf"+RDI_folder_suffix+".fits")
    exit()
    if 1: # 1D extraction
        from breads.instruments.jwstnirspec_cal import fitpsf

        with fits.open(dataobj.webbpsf_filename) as hdulist:
            wpsfs = hdulist[0].data
            wpsfs_header = hdulist[0].header
            wepsfs = hdulist[1].data #*(wpsfs_header["PIXELSCL"]/wpsfs_header["oversamp"])**2
            peak_webb_epsf = np.nanmax(wepsfs, axis=(1, 2))
            wepsfs = wepsfs / peak_webb_epsf[:, None, None]
            wepsfs = wepsfs * dataobj_list[0].aper_to_epsf_peak_f(wv_sampling)[:, None, None]
            webbpsf_wvs = hdulist[2].data
            webbpsf_X = hdulist[3].data
            webbpsf_Y = hdulist[4].data
            webbpsf_X = np.tile(webbpsf_X[None,:,:],(wepsfs.shape[0],1,1))
            webbpsf_Y = np.tile(webbpsf_Y[None,:,:],(wepsfs.shape[0],1,1))

        ann_width = None
        padding = 0.0
        sector_area = None
        fitpsf(dataobj_list,wepsfs,webbpsf_X,webbpsf_Y, out_filename=fitpsf_filename,load=False,IWA = 0.0,OWA = 0.5,#0.5
               mppool=mypool,init_centroid=coords_offset,run_init=False,ann_width=ann_width,padding=padding,
               sector_area=sector_area,RDI_folder_suffix=RDI_folder_suffix,rotate_psf=dataobj_list[0].east2V2_deg,flipx=True,psf_spaxel_area=dataobj_list[0].webbpsf_spaxel_area)
        exit()
    if 0: # run matched filter
        ra_offset = -1.38 # ra offset in as
        dec_offset = -0.92 # dec offset in as
        # ra_vec = np.arange(ra_offset-0.2,ra_offset+0.3,0.2)
        # dec_vec = np.arange(dec_offset-0.2,dec_offset+0.3,0.2)
        ra_vec = np.arange(-2,2,0.1)
        dec_vec = np.arange(-2,2,0.1)

        splitbasename = os.path.basename(filelist[0]).split("_")
        mf_filename = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+"_mf.fits")

        from breads.instruments.jwstnirpsec_cal import matchedfilter
        with fits.open(dataobj.webbpsf_filename) as hdulist:
            wpsfs = hdulist[0].data
            wpsfs_header = hdulist[0].header
            wepsfs = hdulist[1].data *(wpsfs_header["PIXELSCL"]/wpsfs_header["oversamp"])**2
            peak_webb_epsf = np.nanmax(wepsfs, axis=(1, 2))
            wepsfs = wepsfs / peak_webb_epsf[:, None, None]
            webbpsf_wvs = hdulist[2].data
            webbpsf_X = hdulist[3].data
            webbpsf_Y = hdulist[4].data
            webbpsf_X = np.tile(webbpsf_X[None,:,:],(wepsfs.shape[0],1,1))
            webbpsf_Y = np.tile(webbpsf_Y[None,:,:],(wepsfs.shape[0],1,1))

        #normalised psf

        if 1: # read and normalize model grid
            # Define planet model grid from BTsettl
            minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
            # with h5py.File(os.path.join(utils_dir,"BT-Settl_M-0.0_a+0.0_3-6um_500-2500K.hdf5"), 'r') as hf:
            with h5py.File(os.path.join(external_dir,"BT-Settl_3-6um_Teff500_1600_logg3.5_5.0_NIRSpec_3-6um.hdf5"), 'r') as hf:
                grid_specs = np.array(hf.get("spec"))
                grid_temps = np.array(hf.get("temps"))
                grid_loggs = np.array(hf.get("loggs"))
                grid_wvs = np.array(hf.get("wvs"))
            grid_dwvs = grid_wvs[1::]-grid_wvs[0:np.size(grid_wvs)-1]
            grid_dwvs = np.insert(grid_dwvs,0,grid_dwvs[0])
            filter_norm = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs))
            Flambda = np.nansum((grid_dwvs*u.um)[None,None,:]*photfilter_f(grid_wvs)[None,None,:]*(grid_specs*u.W*u.m**-2/u.um),axis=2)/filter_norm
            Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
            grid_specs = grid_specs/Fnu[:,:,None].to(u.MJy).value
            myinterpgrid = RegularGridInterpolator((grid_temps,grid_loggs),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
            rv = 0
            planet_f = interp1d(grid_wvs,myinterpgrid((1000,4.0)), bounds_error=False, fill_value=0)

        #run matched filter
        snr_map,flux_map,fluxerr_map,ra_grid,dec_grid = matchedfilter(fitpsf_filename, dataobj_list, wepsfs,webbpsf_X,webbpsf_Y,
                                                     ra_vec, dec_vec, planet_f,out_filename=mf_filename,linear_interp=True,
                                                     mppool=mypool,rv=rv)#mypool

        #plot snr, flux, err maps
        plt.subplot(1,3,1)
        plt.imshow(snr_map,origin="lower")
        plt.subplot(1,3,2)
        plt.imshow(flux_map,origin="lower")
        plt.subplot(1,3,3)
        plt.imshow(fluxerr_map,origin="lower")
        plt.show()

    if 0: # make cubes
        import subprocess
        import jwst
        from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline
        oversample = True
        RDI_psfsub_dir = os.path.join(os.path.dirname(out_filename), "RDI_psfsub"+RDI_folder_suffix)
        RDI_model_dir = os.path.join(os.path.dirname(out_filename), "RDI_model"+RDI_folder_suffix)
        for dir2reduce in [RDI_psfsub_dir,RDI_model_dir]:
            sstring = os.path.join(dir2reduce, "jw0141400*001_02101_*_" + "*" + "_cal.fits")
            filelist = sorted(glob(sstring))
            filelist.sort()
            print(filelist)
            obs = "psffit"+RDI_folder_suffix


            if oversample:
                spec3_dir = os.path.join(dir2reduce, "step3_cube_0.01as")  # Spec2 pipeline outputs will go here
            else:
                spec3_dir = os.path.join(dir2reduce, "step3_cube_0.1as")  # Spec2 pipeline outputs will go here
            if not os.path.exists(spec3_dir):
                os.makedirs(spec3_dir)

            ## create association file for dithered data. Make a json file for the dither set.
            subprocess.call(["asn_from_list", "-o", f"l3_asn_{obs}.json"] + filelist + ["--product-name", "dither"])

            # This initial setup is just to make sure that we get the latest parameter reference files
            # pulled in for our files.  This is a temporary workaround to get around an issue with
            # how this pipeline calling method works.
            #
            # Setting up steps and running the Spec3 portion of the pipeline. Outlier detector is skipped
            # because it tends to cut high value pixels even if it is a part of the target.

            json_file = f'l3_asn_{obs}.json'
            crds_config = jwst.pipeline.Spec3Pipeline.get_config_from_reference(
                json_file)  # The exact asn file used doesn't matter
            spec3 = jwst.pipeline.Spec3Pipeline.from_config_section(crds_config)
            spec3.output_dir = spec3_dir
            spec3.master_background.skip = False
            spec3.cube_build.skip = False

            # Stephan Birkmann steps
            spec3.cube_build.coord_system = 'ifualign'
            spec3.cube_build.weighting = 'drizzle'
            if oversample:
                spec3.cube_build.scale1 = 0.033
                spec3.cube_build.scale2 = 0.033
                spec3.cube_build.pixfrac = 0.8  # also tried 0.7 and 1.0, smaller kernels deliver 'sharper' images, but underfilling becomes an issue
            spec3.cube_build.single = False

            if filter == 'f290lp':
                spec3.cube_build.wavemin = 2.860
            elif filter == 'f170lp':
                spec3.cube_build.wavemin = 1.654
            elif filter == 'f100lp':
                spec3.cube_build.wavemin = 0.966

            spec3.outlier_detection.save_intermediate_results = False
            spec3.outlier_detection.skip = True  # important, otherwise no good results at the moment
            spec3.outlier_detection.coord_system = 'ifualign'
            # end of Stephan Birkmann steps

            # used on MRS or other modes
            spec3.mrs_imatch.skip = True
            spec3.resample_spec.skip = False
            spec3.combine_1d.skip = True

            # save results
            spec3.save_results = True

            spec3(json_file)

        exit()

    if 1: # combine exp for star_func
        from breads.instruments.jwstnirspec_cal import get_contnorm_spec
        contnorm_filename = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+"_starspec_contnorm.fits")
        spec_R_sampling = 4*dataobj.R
        new_wavelengths,combined_fluxes,combined_errors = \
            get_contnorm_spec(dataobj_list,out_filename=contnorm_filename,load_utils=True,mppool=mypool,spec_R_sampling=spec_R_sampling)
        star_func = interp1d(new_wavelengths, combined_fluxes, kind="cubic", bounds_error=False, fill_value=1)
        starerr_func = interp1d(new_wavelengths, combined_errors, kind="cubic", bounds_error=False, fill_value=1)
        # plt.subplot(2,1,1)
        # plt.plot(new_wavelengths, combined_fluxes)
        # plt.fill_between(new_wavelengths, combined_fluxes-combined_errors, combined_fluxes+combined_errors,alpha=0.5)
        # plt.subplot(2,1,2)
        # plt.plot(new_wavelengths, combined_fluxes/combined_errors)
        # plt.show()
        # exit()

    if 1: # plot spectrum
        if detector == "nrs1":
            fitpsf_filename_nrs1 = fitpsf_filename
            fitpsf_filename_nrs2 = fitpsf_filename.replace("nrs1","nrs2")
        else:
            fitpsf_filename_nrs1 = fitpsf_filename.replace("nrs2","nrs1")
            fitpsf_filename_nrs2 = fitpsf_filename
        flux_calib_wvs = []
        flux_calib = []
        flux2save_wvs = []
        flux2save = []
        photfilter_name_nrs1,wv_sampling_nrs1 = "F360M", np.arange(2.859509, 4.1012874, 0.0006763935)
        photfilter_name_nrs2,wv_sampling_nrs2 = "F460M",np.arange(4.081285,5.278689,0.0006656647)#"F430M"#"F460M"#"F410M"#
        # for det_id,(_detector,_fitpsf_filename,wv_sampling) in enumerate(zip(["nrs1"],[fitpsf_filename_nrs1,fitpsf_filename_nrs2],[wv_sampling_nrs1,wv_sampling_nrs2])):
        for det_id,(_detector,_fitpsf_filename,wv_sampling) in enumerate(zip(["nrs1","nrs2"],[fitpsf_filename_nrs1,fitpsf_filename_nrs2],[wv_sampling_nrs1,wv_sampling_nrs2])):
            with fits.open(_fitpsf_filename) as hdulist:
                bestfit_coords = hdulist[0].data
                wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
                wpsf_ra_offset = hdulist[0].header["INIT_RA"]
                wpsf_dec_offset = hdulist[0].header["INIT_DEC"]
                all_interp_psfsub = hdulist[1].data
                all_interp_psfmodel = hdulist[2].data

            from scipy.signal import medfilt
            norm_fixed_cen = bestfit_coords[:,:, 0] / star_func(wv_sampling)[None,:]
            norm_free_cen = bestfit_coords[:,:, 1] / star_func(wv_sampling)[None,:]
            # norm_fixed_cen = bestfit_coords[:, 0] / star_func(wv_sampling)
            # norm_free_cen = bestfit_coords[:, 1] / star_func(wv_sampling)

            # window_size = 51
            # norm_fixed_cen_LPF = medfilt(norm_fixed_cen, window_size)  # Apply median filtering
            # norm_fixed_cen_HPF = norm_fixed_cen - norm_fixed_cen_LPF  # Subtract median filtered signal from original signal
            # norm_free_cen_LPF = medfilt(norm_free_cen, window_size)  # Apply median filtering
            # norm_free_cen_HPF = norm_free_cen - norm_free_cen_LPF  # Subtract median filtered signal from original signal
            #
            # from scipy.ndimage import generic_filter
            # norm_fixed_cen_std = generic_filter(norm_fixed_cen_HPF, np.nanstd, size=window_size)
            # norm_free_cen_std = generic_filter(norm_free_cen_HPF, np.nanstd, size=window_size)

            # return filtered_signal



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
            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9*np.polyval([-0.03128495, 1.09007397], wv_sampling),linestyle="-.",color=color_list[1],label="Fixed centroid - Corrected",linewidth=2)
            # plt.plot(wv_sampling,bestfit_coords[1,:,0],label="1 Point cloud & WebbPSF (fixed centroid)")
            # plt.plot(wv_sampling,bestfit_coords[1,:,1],label="1 Point cloud & WebbPSF (free centroid)")
            # plt.plot(wv_sampling,bestfit_coords[:,0],label="Point cloud & WebbPSF (fixed centroid)")
            # plt.plot(wv_sampling,bestfit_coords[:,1],label="Point cloud & WebbPSF (free centroid)")
            # plt.plot(ERS_wavelengths, ERS_fluxes_MJy*1e9-0.25,color="grey",linewidth=1, label="Updated Miles+2023",dashes=[5, 3, 10, 3])#,linestyle="-."
            # plt.plot(ERS_wavelengths_v2, ERS_fluxes_MJy_v2*1e-6,color="pink",label="Miles reprocessed")
            plt.plot(stis_wvs,stis_spec_Fnu*1e9,linestyle=":",color="black",label="CALSPEC",linewidth=2)
            plt.xlim([wv_sampling[0],wv_sampling[-1]])
            # plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("Flux density (mJy)",fontsize=fontsize)
            # plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            # plt.subplot(2,1,det_id+1)
            # plt.plot(wv_sampling,norm_free_cen_LPF/norm_fixed_cen_std,label="fixed centroid")
            # plt.plot(wv_sampling,norm_free_cen_LPF/norm_free_cen_std,label="free centroid")
            if det_id == 0:
                plt.ylim(4.5,10.5)
                plt.legend(loc="upper right")
                plt.text(0.01,0.01, '4 dithers - TYC 4433-1800-1', fontsize=fontsize, ha='left', va='bottom',color="black", transform=plt.gca().transAxes)# \n 36-166 $\mu$Jy
            if det_id == 1:
                plt.ylim(3.0,5.5)
            plt.xticks([])

            ax1 = plt.subplot(gs[3*det_id+det_id+1, 0])
            plt.plot(ERS_wavelengths, ERS_fluxes_MJy*1e9,color=color_list[0],linewidth=2, label="Updated Miles+2023",dashes=[5, 3, 10, 3])#,linestyle="-."
            # plt.plot(ERS_wavelengths_v2, ERS_fluxes_MJy_v2*1e-6,color="pink",label="Miles reprocessed")
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
            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9*np.polyval([-0.03128495, 1.09007397], wv_sampling)-calspec_func(wv_sampling),linestyle="-.",color=color_list[1],label="Fixed centroid - Corrected",linewidth=1)
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
            plt.plot(wv_sampling,bestfit_coords[0,:,0]/bestfit_coords[0,:,1])
            plt.subplot(2,2,2)
            plt.title("Ratios with Calspec")
            interp_stis_spec = interp1d(stis_wvs,stis_spec_Fnu,bounds_error=False,fill_value=np.nan)(wv_sampling)
            plt.plot(wv_sampling,interp_stis_spec/bestfit_coords[0,:,0])
            plt.plot(wv_sampling,bestfit_coords[0,:,0]/interp_stis_spec)
            plt.plot(wv_sampling,interp_stis_spec/bestfit_coords[0,:,1],linestyle="--")
            plt.plot(wv_sampling,bestfit_coords[0,:,1]/interp_stis_spec,linestyle="--")
            flux_calib_wvs.extend(wv_sampling)
            flux_calib.extend(interp_stis_spec / bestfit_coords[0, :, 0])
            plt.subplot(2,2,3)
            plt.title("displacement")
            plt.plot(wv_sampling,np.sqrt((bestfit_coords[0,:,2]-(-0.256))**2+(bestfit_coords[0,:,3]-(0.751))**2))

        flux_calib_wvs,flux_calib = np.array(flux_calib_wvs),np.array(flux_calib)
        wherefinite = np.where(np.isfinite(flux_calib))
        flux_calib_wvs, flux_calib=flux_calib_wvs[wherefinite],flux_calib[wherefinite]
        print("flux calibration", np.polyfit(flux_calib_wvs,flux_calib ,deg=1))


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