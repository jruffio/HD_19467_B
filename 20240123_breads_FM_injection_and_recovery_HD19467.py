import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
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

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    ####################
    ## To be modified
    ####################
    fontsize = 12
    # output dir for images
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
    # out_png = out_dir.replace("xy","figures")
    if not os.path.exists(out_png):
        os.makedirs(out_png)
    # Number of threads to be used for multithreading
    numthreads = 20
    # Number of nodes
    nodes = 40
    # Number of principal components (Karhunen-Loeve) modes to be added to the forward model
    N_KL = 3#0#3
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/external/"
    # Science data: List of stage 2 cal.fits files
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    # utility folder where the intermediate and final data product will be saved
    utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils_fm/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    # out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_out_fm/xy/"
    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240216_out_fm/xy/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # spectrum to be used for the companion template
    RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/HD19467b_RDI_1dspectrum_MJy.fits"
    ####################
    ## No need to change for HD19467B prog. id 1414
    # NIrcam filter used for flux normalization
    photfilter_name_nrs1 =  "F360M"
    photfilter_name_nrs2 =  "F460M"
    # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
    wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
    wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
    # definiton of the nodes for the spline
    x_nodes_nrs1 = np.linspace(2.859509, 4.1012874, nodes, endpoint=True)
    x_nodes_nrs2 = np.linspace(4.081285, 5.278689, nodes, endpoint=True)
    # if 1: # Different attempt at define the nodes
    #     dw1,dw2 = 0.02,0.04
    #     l0,l2 = 2.859509-dw1, 4.1012874
    #     l1 = (l0+l2)/2
    #     x1_nodes = np.arange(l0,l1, dw1)
    #     l1 = x1_nodes[-1]+dw2
    #     x2_nodes = np.arange(l1,l2, dw2)
    #     x_nodes_nrs1 = np.concatenate([x1_nodes,x2_nodes])
    # if 1:
    #     dw1,dw2 = 0.04,0.02
    #     l0,l2 = 4.081285,5.278689+dw2
    #     l1 = (l0+l2)/2
    #     x1_nodes = np.arange(l0,l1, dw1)
    #     l1 = x1_nodes[-1]+dw2
    #     x2_nodes = np.arange(l1,l2, dw2)
    #     x_nodes_nrs2 = np.concatenate([x1_nodes,x2_nodes])
    # List of empirical offsets to correct the wcs coordinate system
    centroid_nrs1 = [-0.13499443, -0.07202978]
    centroid_nrs2 = [-0.12986898, -0.08441719]
    # RA Dec offset of the companion HD19467B
    ra_offset = -1332.871/1000. # ra offset in as
    dec_offset = -875.528/1000. # dec offset in as
        # Offsets for HD 19467 B from https://www.whereistheplanet.com/
        # RA Offset = -1332.871 +/- 10.886 mas
        # Dec Offset = -875.528 +/- 12.360 mas
        # Separation = 1593.703 +/- 9.530 mas
        # PA = 236.712 +/- 0.483 deg
        # Reference: Brandt et al. 2021
    # Absolute fluxes for the host star to be used in calculated flux ratios with the companion.
    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}
    # Flux Calibration parameters
    flux_calib_paras = [-0.03864459,  1.09360589]
    ####################

    if 0: # compute and save things
        mypool = mp.Pool(processes=numthreads)

        for filename in filelist[0:2]:
            print(filename)

            if "nrs1" in filename:
                # continue
                detector = "nrs1"
                wv_sampling = wv_sampling_nrs1
                photfilter_name = photfilter_name_nrs1
                centroid = centroid_nrs1
                x_nodes = x_nodes_nrs1
            elif "nrs2" in filename:
                # continue
                detector = "nrs2"
                wv_sampling = wv_sampling_nrs2
                photfilter_name = photfilter_name_nrs2
                centroid = centroid_nrs2
                x_nodes = x_nodes_nrs2

            try:
                photfilter = os.path.join(external_dir, "JWST_NIRCam." + photfilter_name + ".dat")
                filter_arr = np.loadtxt(photfilter)
            except:
                photfilter = os.path.join(external_dir, "Keck_NIRC2." + photfilter_name + ".dat")
                filter_arr = np.loadtxt(photfilter)
            trans_wvs = filter_arr[:, 0] / 1e4
            trans = filter_arr[:, 1]
            photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
            photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
            bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
            photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]
            print(photfilter_wvmin, photfilter_wvmax)

            preproc_task_list = []
            preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
            preproc_task_list.append(["compute_coordinates_arrays"])
            preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
            preproc_task_list.append(["apply_coords_offset",{"coords_offset": centroid}])
            preproc_task_list.append(["compute_quick_webbpsf_model", {"image_mask": None, "pixelscale": 0.1, "oversample": 10}, True, True])
            # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
            preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
            preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": x_nodes, "mppool": mypool}, True, True])
            preproc_task_list.append(["compute_starsubtraction", {"mppool": mypool}, True, True])
            # preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True])

            dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                      save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
            # plt.imshow(dataobj.area2d,origin="lower")
            # plt.colorbar()
            # plt.show()
            from breads.instruments.jwstnirspec_cal import where_point_source
            where2mask = where_point_source(dataobj,(ra_offset,dec_offset),0.4)
            tmp_badpixels = copy(dataobj.bad_pixels)
            tmp_badpixels[where2mask] = np.nan
            subtracted_im,star_model,spline_paras,_ = dataobj.reload_starsubtraction()

            if 1:
                from breads.instruments.jwstnirspec_cal import PCA_wvs_axis
                ny,nx = dataobj.data.shape
                first_half = np.where(dataobj.wavelengths<wv_sampling[np.size(wv_sampling)//2])
                second_half = np.where(dataobj.wavelengths>wv_sampling[np.size(wv_sampling)//2])
                wv4pca,im4pcs,n4pca,bp4pca =  copy(dataobj.wavelengths),copy(subtracted_im),copy(dataobj.noise),copy(tmp_badpixels)
                # wv4pca[second_half] = np.nan
                # im4pcs[second_half] = np.nan
                # n4pca[second_half] = np.nan
                bp4pca[second_half] = np.nan
                KLs_wvs_left,KLs_left = PCA_wvs_axis(wv4pca,im4pcs,n4pca,bp4pca,
                                   np.nanmedian(dataobj.wavelengths)/(4*dataobj.R),N_KL=N_KL)
                wv4pca,im4pcs,n4pca,bp4pca =  copy(dataobj.wavelengths),copy(subtracted_im),copy(dataobj.noise),copy(tmp_badpixels)
                # wv4pca[first_half] = np.nan
                # im4pcs[first_half] = np.nan
                # n4pca[first_half] = np.nan
                bp4pca[first_half] = np.nan
                KLs_wvs_right,KLs_right = PCA_wvs_axis(wv4pca,im4pcs,n4pca,bp4pca,
                                   np.nanmedian(dataobj.wavelengths)/(4*dataobj.R),N_KL=N_KL)
                wv4pca,im4pcs,n4pca,bp4pca =  copy(dataobj.wavelengths),copy(subtracted_im),copy(dataobj.noise),copy(tmp_badpixels)
                KLs_wvs_all,KLs_all = PCA_wvs_axis(wv4pca,im4pcs,n4pca,bp4pca,
                                   np.nanmedian(dataobj.wavelengths)/(4*dataobj.R),N_KL=N_KL)
                wvs_KLs_f_list = []
                for k in range(KLs_left.shape[1]):
                    KL_f = interp1d(KLs_wvs_left,KLs_left[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
                    wvs_KLs_f_list.append(KL_f)
                for k in range(KLs_right.shape[1]):
                    KL_f = interp1d(KLs_wvs_right,KLs_right[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
                    wvs_KLs_f_list.append(KL_f)
            else:
                wvs_KLs_f_list = None
            # plt.figure(2)
            # for k in range(KLs_left.shape[1]):
            #     plt.subplot(KLs_left.shape[1],1,k+1)
            #     plt.plot(KLs_wvs_left,KLs_left[:,k],label="{0}".format(k))
            # plt.figure(3)
            # for k in range(KLs_right.shape[1]):
            #     plt.subplot(KLs_right.shape[1],1,k+1)
            #     plt.plot(KLs_wvs_right,KLs_right[:,k],label="{0}".format(k))
            # plt.figure(4)
            # for k in range(KLs_all.shape[1]):
            #     plt.subplot(KLs_all.shape[1],1,k+1)
            #     plt.plot(KLs_wvs_all,KLs_all[:,k],label="{0}".format(k))
            # # plt.legend()
            # plt.show()

            if 1: # read and normalize companion template used in FM
                hdulist_sc = fits.open(RDI_spec_filename)
                grid_wvs = hdulist_sc[0].data
                RDI_spec = ((hdulist_sc[1].data*u.MJy)*(const.c/(grid_wvs*u.um)**2)).to(u.W*u.m**-2/u.um).value
                err = hdulist_sc[2].data

                # add flux calibration correction because we are fitting a pure WebbPSF model later, which comes with systematics
                RDI_spec = RDI_spec/np.polyval(flux_calib_paras, grid_wvs)

                grid_dwvs = grid_wvs[1::]-grid_wvs[0:np.size(grid_wvs)-1]
                grid_dwvs = np.insert(grid_dwvs,0,grid_dwvs[0])
                filter_norm = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs))
                Flambda = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs)*(RDI_spec*u.W*u.m**-2/u.um))/filter_norm
                Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
                RDI_spec = RDI_spec/Fnu.to(u.MJy).value

                # The code expects a model grid but we have a single spectrum, so we are creating a dumb grid but fixing the parameters anyway...
                grid_specs = np.tile(RDI_spec[None,None,:],(2,2,1))
                # grid_specs[np.where(np.isnan(grid_specs))] = 0
                myinterpgrid = RegularGridInterpolator(([-1,1],[-1,1]),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
                teff,logg,vsini,rv,dra_comp,ddec_comp = 0.0,0.0,0.0,None,None,None
                fix_parameters = [teff,logg,vsini,rv,dra_comp,ddec_comp]

            if 1: # Definition of the priors
                subtracted_im,star_model,spline_paras,x_nodes = dataobj.reload_starsubtraction()

                wherenan = np.where(np.isnan(spline_paras))
                reg_mean_map = copy(spline_paras)
                reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras, axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
                if 0:
                    reg_std_map = np.zeros(spline_paras.shape)+300e-12#np.nanmedian(spline_paras)
                else:
                    reg_std_map = np.abs(spline_paras)
                    reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras), axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
                    # reg_std_map[wherenan] = np.tile(np.zeros(spline_paras.shape), (1, spline_paras.shape[1]))[wherenan]
                    reg_std_map = reg_std_map
                    reg_std_map = np.clip(reg_std_map, 1e-11, np.inf) # in MJy

            from breads.fm.hc_atmgrid_splinefm_jwst_nirspec_cal import hc_atmgrid_splinefm_jwst_nirspec_cal
            fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                        "radius_as":0.1,"badpixfraction":0.75,"nodes":x_nodes,"fix_parameters":fix_parameters,
                        "wvs_KLs_f":wvs_KLs_f_list,"regularization":"user","reg_mean_map":reg_mean_map,"reg_std_map":reg_std_map}
            fm_func = hc_atmgrid_splinefm_jwst_nirspec_cal


            outoftheoven_filelist = glob(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
            outoftheoven_filelist.sort()
            outoftheoven_filename = outoftheoven_filelist[-1]
            with h5py.File(outoftheoven_filename, 'r') as hf:
                rvs = np.array(hf.get("rvs"))
                ras = np.array(hf.get("ras"))
                decs = np.array(hf.get("decs"))
                log_prob = np.array(hf.get("log_prob"))
                log_prob_H0 = np.array(hf.get("log_prob_H0"))
                rchi2 = np.array(hf.get("rchi2"))
                linparas = np.array(hf.get("linparas"))
                linparas_err = np.array(hf.get("linparas_err"))
            linparas_err = np.swapaxes(linparas_err, 1, 2)
            err_map = linparas_err[0,:,:,0]
            dra=ras[1]-ras[0]
            ddec=decs[1]-decs[0]
            ras_grid,decs_grid = np.meshgrid(ras,decs)
            rs_grid = np.sqrt(ras_grid**2+decs_grid**2)
            rs_comp_grid = np.sqrt((ras_grid-ra_offset)**2+(decs_grid-dec_offset)**2)

            sep_fake_list = np.arange(0.3,3.0,0.1)
            pa_fake_list = 45*np.arange(np.size(sep_fake_list))
            ra_fake_list = sep_fake_list * np.sin(np.deg2rad(pa_fake_list))
            dec_fake_list = sep_fake_list * np.cos(np.deg2rad(pa_fake_list))

            original_dataobjdata = copy(dataobj.data)

            flux_fake_list = np.zeros(np.size(sep_fake_list))+np.nan
            recovered_flux_fake_list = np.zeros(np.size(sep_fake_list))+np.nan
            recovered_fluxerr_fake_list = np.zeros(np.size(sep_fake_list))+np.nan
            for fakeid, (sep_fake,pa_fake,ra_fake,dec_fake) in enumerate(zip(sep_fake_list,pa_fake_list,ra_fake_list,dec_fake_list)):
                dist_fake2comp = np.sqrt((ra_offset-ra_fake)**2+(dec_offset-dec_fake)**2)
                dist2fake = np.sqrt((ras_grid-ra_fake)**2+(decs_grid-dec_fake)**2)
                k_fake,l_fake = np.unravel_index(np.argmin(dist2fake),dist2fake.shape)
                # print(k_fake,l_fake)
                # plt.imshow(err_map,origin="lower")
                # plt.show()
                flux_fake_list[fakeid] = err_map[k_fake,l_fake]*10
                if flux_fake_list[fakeid] == np.nan or dist_fake2comp < 0.5:
                    continue
                print(flux_fake_list[fakeid],sep_fake,pa_fake,ra_fake,dec_fake)

                # # plt.scatter(ra_fake_list,dec_fake_list)
                # plt.plot(sep_fake_list,flux_fake_list)
                # plt.show()
                # #inject fake planet in reduction for contrast curve calibration
                dataobj.data = copy(original_dataobjdata)
                fm_paras_tmp = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                                "radius_as":0.5,"badpixfraction":0.99,"nodes":x_nodes,"fix_parameters":fix_parameters,
                                "Nrows_max":500}
                nonlin_paras = [0, ra_fake,dec_fake]
                try:
                    d, M, s,extra_outputs = fm_func(nonlin_paras,dataobj,return_extra_outputs=True,**fm_paras_tmp)
                except:
                    continue
                where_finite = extra_outputs["where_trace_finite"]
                if detector == "nrs1":
                    dataobj.data[where_finite] += flux_fake_list[fakeid] * M[:,0]
                elif detector == "nrs2":
                    # dataobj.data[where_finite] += 2*2e-12 * M[:,0]
                    dataobj.data[where_finite] += flux_fake_list[fakeid] * M[:,0]

                nonlin_paras = [0.0,ra_fake,dec_fake] # rv (km/s), dra (arcsec),ddec (arcsec),
                # d is the data vector a the specified location
                # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
                # s is the vector of uncertainties corresponding to d
                try:
                    d, M, s,extra_outputs = fm_func(nonlin_paras,dataobj,return_extra_outputs=True,**fm_paras)
                except:
                    continue
                where_finite = extra_outputs["where_trace_finite"]
                w = extra_outputs["wvs"]
                x = extra_outputs["ras"]
                y = extra_outputs["decs"]
                rows = extra_outputs["rows"]
                d_reg,s_reg = extra_outputs["regularization"]
                reg_wvs = extra_outputs["regularization_wvs"]
                reg_rows = extra_outputs["regularization_rows"]
                unique_rows = np.unique(rows)
                shifted_w = w+(wv_sampling[-1]-wv_sampling[0])*(rows-np.nanmin(unique_rows))
                shifted_reg_wvs = reg_wvs+(wv_sampling[-1]-wv_sampling[0])*(reg_rows-np.nanmin(unique_rows))

                # print(np.unique(where_finite[0]))
                # mask_comp = np.zeros(dataobj.data.shape)
                # mask_comp[where_finite] = 1


                validpara = np.where(np.max(np.abs(M),axis=0)!=0)
                M = M[:,validpara[0]]

                d = d / s
                M = M / s[:, None]

                from breads.fit import fitfm
                residuals = np.ones(np.size(s)+np.size(validpara[0]))+np.nan
                residuals_H0 = np.ones(np.size(s)+np.size(validpara[0]))+np.nan
                noise4residuals = np.ones(np.size(s)+np.size(validpara[0]))+np.nan
                log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,
                                                                             computeH0 = True,bounds = None,
                                                                             residuals=residuals,residuals_H0=residuals_H0,noise4residuals=noise4residuals,
                                                                             scale_noise=True,marginalize_noise_scaling=False)
                recovered_flux_fake_list[fakeid] = linparas[0]
                recovered_fluxerr_fake_list[fakeid] = linparas_err[0]
                print(linparas[0],linparas_err[0],flux_fake_list[fakeid])
                print(recovered_flux_fake_list)
                print(recovered_fluxerr_fake_list)


            out_filename = os.path.join(out_png, "injection_and_recovery_"+detector+".h5")
            print(out_filename)
            with h5py.File(out_filename, 'w') as hf:
                hf.create_dataset("sep_fake_list", data=sep_fake_list)
                hf.create_dataset("pa_fake_list", data=pa_fake_list)
                hf.create_dataset("ra_fake_list", data=ra_fake_list)
                hf.create_dataset("dec_fake_list", data=dec_fake_list)
                hf.create_dataset("flux_fake_list", data=flux_fake_list)
                hf.create_dataset("recovered_flux_fake_list", data=recovered_flux_fake_list)
                hf.create_dataset("recovered_fluxerr_fake_list", data=recovered_fluxerr_fake_list)
        mypool.close()
        mypool.join()

    if 1: # Load and plot things
        plt.figure(1,figsize=(6,4.0))
        # for detector in ["nrs2"]:
        for detector,marker in zip(["nrs1","nrs2"],["x","o"]):
            out_filename = os.path.join(out_png, "injection_and_recovery_" + detector + ".h5")
            with h5py.File(out_filename, 'r') as hf:
                sep_fake_list = np.array(hf.get("sep_fake_list"))
                pa_fake_list = np.array(hf.get("pa_fake_list"))
                ra_fake_list = np.array(hf.get("ra_fake_list"))
                dec_fake_list = np.array(hf.get("dec_fake_list"))
                flux_fake_list = np.array(hf.get("flux_fake_list"))
                recovered_flux_fake_list = np.array(hf.get("recovered_flux_fake_list"))
                recovered_fluxerr_fake_list = np.array(hf.get("recovered_fluxerr_fake_list"))

            print(recovered_flux_fake_list/flux_fake_list)
            plt.scatter(sep_fake_list,recovered_flux_fake_list/flux_fake_list,label=detector,marker=marker)
            plt.errorbar(sep_fake_list,recovered_flux_fake_list/flux_fake_list,yerr=[recovered_fluxerr_fake_list/flux_fake_list], ls='none')
        plt.ylim([0.8,1.2])
        plt.xlabel("Separation (arcsec)",fontsize=fontsize)
        plt.ylabel(r"Flux ratio (recovered/injected)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper left", handlelength=3,fontsize=fontsize)

        plt.tight_layout()

        out_filename = os.path.join(out_png, "injection_and_recovery_HD19467.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

        plt.show()


    exit()