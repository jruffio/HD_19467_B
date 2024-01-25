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
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"

    numthreads = 20
    N_KL = 3#0#3
    nodes = 40#40
    mygrid="RDIspec"

    # RA Dec offset of the companion
    ra_offset = -1332.871/1000. # ra offset in as
    dec_offset = -875.528/1000. # dec offset in as
    # ra_offset = -1.38 # ra offset in as
    # dec_offset = -0.92 # dec offset in as
    # Offsets for HD 19467 B
    # RA Offset = -1332.871 +/- 10.886 mas
    # Dec Offset = -875.528 +/- 12.360 mas
    # Separation = 1593.703 +/- 9.530 mas
    # PA = 236.712 +/- 0.483 deg
    # Reference: Brandt et al. 2021

    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240123_out_kl{0}_nodes{1}_{2}/xy/".format(N_KL,nodes,mygrid)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240123_utils_nodes{1}/".format(N_KL,nodes,mygrid)
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"

    # Choose to reduce either detector (NRS1 or NRS2)
    # detector,photfilter_name,wv_sampling,x_knots = "nrs1", "F360M", np.arange(2.859509, 4.1012874, 0.0006763935),np.arange(2.859509-0.03,4.1012874+0.03,0.06)
    detector,photfilter_name,wv_sampling,x_knots = "nrs2", "F460M",np.arange(4.081285,5.278689,0.0006656647),np.arange(4.081285-0.03,5.278689+0.03,0.03)

    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2/jw01414004001_02101_*_"+detector+"_cal.fits")
    filelist.sort()
    print("I found the following files:")
    for fid,filename in enumerate(filelist):
        print(filename)
    print("N files: {0}".format(len(filelist)))

    # Define
    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}

    try:
        photfilter = os.path.join(external_dir,"JWST_NIRCam."+photfilter_name+".dat")
        filter_arr = np.loadtxt(photfilter)
    except:
        photfilter = os.path.join(external_dir,"Keck_NIRC2."+photfilter_name+".dat")
        filter_arr = np.loadtxt(photfilter)
    trans_wvs = filter_arr[:,0]/1e4
    trans = filter_arr[:,1]
    photfilter_f = interp1d(trans_wvs,trans,bounds_error=False,fill_value=0)
    photfilter_wv0 = np.nansum(trans_wvs*photfilter_f(trans_wvs))/np.nansum(photfilter_f(trans_wvs))
    bandpass = np.where(photfilter_f(trans_wvs)/np.nanmax(photfilter_f(trans_wvs))>0.01)
    photfilter_wvmin,photfilter_wvmax = trans_wvs[bandpass[0][0]],trans_wvs[bandpass[0][-1]]
    print(photfilter_wvmin,photfilter_wvmax)
    # exit()
    # plt.plot(trans_wvs,trans)
    # plt.show()

    mypool = mp.Pool(processes=numthreads)

    preproc_task_list = []
    preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
    preproc_task_list.append(["compute_coordinates_arrays"])
    # preproc_task_list.append(["convert_MJy_per_sr_to_MJy"]) # old reduction, already in MJy
    ra_corr,dec_corr = -0.1285876002234175, - 0.06997868615326872
    preproc_task_list.append(["apply_coords_offset",{"coords_offset":(ra_corr,dec_corr)}]) #-0.11584366936455087, 0.07189009712128012
    preproc_task_list.append(["compute_webbpsf_model",
                              {"wv_sampling": wv_sampling, "image_mask": None, "pixelscale": 0.1, "oversample": 10,
                               "parallelize": False, "mppool": mypool}, True, True])
    # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
    preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
    preproc_task_list.append(["compute_starspectrum_contnorm", {"x_knots": x_knots, "mppool": mypool}, True, True])

    # if 1:
    #     filename = filelist[0]
    for filename in filelist[0::]:
        print(filename)
        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
        exit()
        # dataobj = JWSTNirspec_cal(filename,crds_dir=crds_dir,utils_dir=utils_dir,save_utils=True, load_utils=True,mppool=mypool,
        #                           regwvs_sampling = wv_sampling,
        #                           load_interpdata_regwvs=True,
        #                           wpsffit_IWA=0.3,wpsffit_OWA=1.0,
        #                           mask_charge_bleeding=True,compute_wpsf=True,compute_starspec_contnorm=True,compute_starsub=True,compute_interp_regwvs=True,fit_wpsf=False,
        #                           apply_chargediff_mask=True)
        # print(dataobj.wpsf_ra_offset,dataobj.wpsf_dec_offset)
        # exit()
    # exit()
    # if 0:
        # dataobj.bad_pixels[np.where((photfilter_wvmin<dataobj.wavelengths)*(dataobj.wavelengths<photfilter_wvmax))] = np.nan

        # if 1:
        #     all_interp_ref0_ra, all_interp_ref0_dec, all_interp_ref0_wvs, all_interp_ref0_flux, all_interp_ref0_err, all_interp_ref0_badpix, all_interp_ref0_area2d = \
        #         dataobj.interpdata_regwvs(wv_sampling=wv_sampling, modelfit=False, out_filename=dataobj.interpdata_regwvs_filename,
        #                                    load_interpdata_regwvs=True)
        #     all_interp_ref0_ra -= dataobj.wpsf_ra_offset
        #     all_interp_ref0_dec -= dataobj.wpsf_dec_offset
        #     R = np.sqrt((dataobj.dra_as_array-dataobj.wpsf_ra_offset)**2+(dataobj.ddec_as_array-dataobj.wpsf_dec_offset)**2)
        #     val_list = []
        #     R_list = []
        #     from breads.instruments.jwstnirpsec_cal import untangle_dq
        #     hdulist_sc = fits.open(dataobj.filename)
        #     priheader = hdulist_sc[0].header
        #     extheader = hdulist_sc[1].header
        #     im = hdulist_sc["SCI"].data
        #     err = hdulist_sc["ERR"].data
        #     dq = hdulist_sc["DQ"].data
        #     DQ_cube = untangle_dq(dq)
        #     print(np.nansum(DQ_cube,axis=(1,2)))
        #     for rowid in range(dataobj.data.shape[0]):
        #         try:
        #             wvid = np.nanargmin(np.abs(dataobj.wavelengths[rowid,:]-3.0))
        #             R_list.append(R[rowid,wvid])
        #             val_list.append(DQ_cube[0,rowid,wvid])
        #         except:
        #             pass
        #     plt.scatter(R_list,val_list)
        #     plt.show()

        from breads.instruments.jwstnirspec_cal import where_point_source
        where2mask = where_point_source(dataobj,(ra_offset,dec_offset),0.4)
        # plt.subplot(1,2,1)
        # plt.imshow(dataobj.bad_pixels,origin="lower")
        tmp_badpixels = copy(dataobj.bad_pixels)
        tmp_badpixels = tmp_badpixels*dataobj.bar_mask
        tmp_badpixels[where2mask] = np.nan
        with fits.open(dataobj.starsub_filename) as hdulist:
            subtracted_im = hdulist[0].data
        # plt.subplot(1,2,2)
        # plt.imshow(tmp_badpixels,origin="lower")
        # plt.show()

        # from breads.instruments.jwstnirpsec_cal import PCA_detec
        # # plt.figure(1)
        # detec_KLs = PCA_detec(subtracted_im, dataobj.noise, tmp_badpixels,N_KL=N_KL)
        # # for k in range(KLs.shape[1]):
        # #     plt.subplot(KLs.shape[1],1,k+1)
        # #     plt.plot(KLs[:, k], label="{0}".format(k))
        # # plt.legend()

        if N_KL != 0:
            # N_KL=10
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
            # wv4pca,im4pcs,n4pca,bp4pca =  copy(dataobj.wavelengths),copy(subtracted_im),copy(dataobj.noise),copy(tmp_badpixels)
            # KLs_wvs_all,KLs_all = PCA_wvs_axis(wv4pca,im4pcs,n4pca,bp4pca,
            #                    np.nanmedian(dataobj.wavelengths)/(4*dataobj.R),N_KL=N_KL)
            wvs_KLs_f_list = []
            for k in range(KLs_left.shape[1]):
                KL_f = interp1d(KLs_wvs_left,KLs_left[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
                wvs_KLs_f_list.append(KL_f)
            for k in range(KLs_right.shape[1]):
                KL_f = interp1d(KLs_wvs_right,KLs_right[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
                wvs_KLs_f_list.append(KL_f)
        else:
            wvs_KLs_f_list = None
        plt.figure(2)
        for k in range(KLs_left.shape[1]):
            plt.subplot(KLs_left.shape[1],1,k+1)
            plt.plot(KLs_wvs_left,KLs_left[:,k],label="{0}".format(k))
        plt.figure(3)
        for k in range(KLs_right.shape[1]):
            plt.subplot(KLs_right.shape[1],1,k+1)
            plt.plot(KLs_wvs_right,KLs_right[:,k],label="{0}".format(k))
        # plt.figure(4)
        # for k in range(KLs_all.shape[1]):
        #     plt.subplot(KLs_all.shape[1],1,k+1)
        #     plt.plot(KLs_wvs_all,KLs_all[:,k],label="{0}".format(k))
        # plt.legend()
        plt.show()

        if mygrid == "RDIspec": # read and normalize model grid
            RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/20230705_HD19467b_RDI_1dspectrum.fits"
            hdulist_sc = fits.open(RDI_spec_filename)
            grid_wvs = hdulist_sc[0].data
            grid_specs = ((hdulist_sc[1].data[None,None,:]*u.MJy)*(const.c/(grid_wvs*u.um)**2)).to(u.W*u.m**-2/u.um).value
            err = hdulist_sc[2].data

            grid_dwvs = grid_wvs[1::]-grid_wvs[0:np.size(grid_wvs)-1]
            grid_dwvs = np.insert(grid_dwvs,0,grid_dwvs[0])
            filter_norm = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs))
            Flambda = np.nansum((grid_dwvs*u.um)[None,None,:]*photfilter_f(grid_wvs)[None,None,:]*(grid_specs*u.W*u.m**-2/u.um),axis=2)/filter_norm
            Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
            grid_specs = grid_specs/Fnu[:,:,None].to(u.MJy).value
            grid_specs = np.tile(grid_specs,(2,2,1))
            # grid_specs[np.where(np.isnan(grid_specs))] = 0
            myinterpgrid = RegularGridInterpolator(([-1,1],[-1,1]),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
            teff,logg,vsini,rv,dra_comp,ddec_comp = 0.0,0.0,0.0,None,None,None
            fix_parameters = [teff,logg,vsini,rv,dra_comp,ddec_comp]
            # plt.plot(grid_wvs,grid_specs[0,0,:])
            # # plt.plot(myinterpgrid((0,0)))
            # # plt.plot(grid_wvs, np.isfinite(myinterpgrid((0,0))).astype(float))
            # plt.show()
            # mask_f = interp1d(grid_wvs, np.isfinite(myinterpgrid((0,0))).astype(float), bounds_error=False, fill_value=0)
            # model_spec = myinterpgrid((0,0))
            # model_spec[np.where(np.isnan(model_spec))] = 0
            # planet_f = interp1d(grid_wvs, myinterpgrid((0,0)), bounds_error=False, fill_value=0)
            # minwv, maxwv=2.8,5.2
            # mywvs=np.linspace(minwv, maxwv, 2000)
            # toplotspec = planet_f(mywvs)
            # # toplotspec[np.where(mask_f(mywvs)!=1)] = np.nan
            # plt.plot(mywvs,toplotspec,color="red")
            # # plt.plot(mywvs,mask_f(mywvs),color="green")
            # plt.show()

            # planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)
        elif mygrid == "BTsettl15": # read and normalize model grid
            # Define planet model grid from BTsettl
            minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
            #/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/BT-Settl15_0.5-6.0um_Teff500_1600_logg3.5_5.0.hdf5
            # with h5py.File(os.path.join(utils_dir,"BT-Settl_M-0.0_a+0.0_3-6um_500-2500K.hdf5"), 'r') as hf:
            with h5py.File(os.path.join(external_dir,"BT-Settl15_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"), 'r') as hf:
            # with h5py.File(os.path.join(external_dir,"BT-Settl_3-6um_Teff500_1600_logg3.5_5.0_NIRSpec_3-6um.hdf5"), 'r') as hf:
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

            # plt.plot(trans_wvs,trans)
            # plt.show()
            # self.webbpsf_im = webb_epsf_ims[psf_wv0_id]/peak_webb_epsf[psf_wv0_id]
            # self.webbpsf_X = webbpsf_X
            # self.webbpsf_Y = webbpsf_Y
            # self.webbpsf_wv0 = webbpsf_wvs[psf_wv0_id]
            # self.webbpsf_interp = CloughTocher2DInterpolator((self.webbpsf_X.flatten(), self.webbpsf_Y.flatten()), self.webbpsf_im.flatten(),fill_value=0.0)

            myinterpgrid = RegularGridInterpolator((grid_temps,grid_loggs),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
            teff,logg,vsini,rv,dra_comp,ddec_comp = 1100,5.0,0.0,None,None,None
            fix_parameters = [teff,logg,vsini,rv,dra_comp,ddec_comp]
        if mygrid == "sonora":
            gridfilename = os.path.join(external_dir,"SonoraCholla2021nc_logKzz2_7_Teff500_1300_g31_3162_NIRSpec_3-6um.hdf5")
            with h5py.File(gridfilename, 'r') as hf:
                grid_specs = np.array(hf.get("spec"))
                grid_metals = np.array(hf.get("logKzzs"))
                grid_temps = np.array(hf.get("temps"))
                grid_loggs = np.array(hf.get("loggs"))
                grid_wvs = np.array(hf.get("wvs"))
            grid_dwvs = grid_wvs[1::]-grid_wvs[0:np.size(grid_wvs)-1]
            grid_dwvs = np.insert(grid_dwvs,0,grid_dwvs[0])

            filter_norm = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs))
            Flambda = np.nansum((grid_dwvs*u.um)[None,None,None,:]*photfilter_f(grid_wvs)[None,None,None,:]*(grid_specs*u.W*u.m**-2/u.um),axis=-1)/filter_norm
            Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
            grid_specs = grid_specs/Fnu[:,:,:,None].to(u.MJy).value
            myinterpgrid = RegularGridInterpolator((grid_metals,grid_temps,grid_loggs),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
            metal,teff,logg,vsini,rv,dra_comp,ddec_comp = 2.0,1100,4.5,0.0,None,None,None
            fix_parameters = [metal,teff,logg,vsini,rv,dra_comp,ddec_comp]



        x_knots = np.linspace(np.nanmin(dataobj.wavelengths), np.nanmax(dataobj.wavelengths), nodes, endpoint=True)

        if 1:
            # with fits.open(dataobj.starsub_filename) as hdulist:
            #     star_model = hdulist[2].data
            # data_smooth = star_model/dataobj.star_func(dataobj.wavelengths)

            reg_mean_map0 = np.zeros((dataobj.data.shape[0], nodes))
            reg_std_map0 = np.zeros((dataobj.data.shape[0], nodes))
            for rowid, row in enumerate(dataobj.data):
                row_wvs = dataobj.wavelengths[rowid, :]
                row_bp = dataobj.bad_pixels[rowid, :]
                if np.nansum(np.isfinite(row * row_bp)) == 0:
                    continue

                reg_mean_map0[rowid, :] = np.nanmedian(row * row_bp)
                reg_std_map0[rowid, :] = reg_mean_map0[rowid, :]

            from breads.instruments.jwstnirspec_cal import normalize_rows

            threshold_badpix = 10
            star_model, _, new_badpixs, subtracted_im, spline_paras = normalize_rows(dataobj.data,
                                                                                      dataobj.wavelengths,
                                                                                      noise=dataobj.noise,
                                                                                      badpixs=dataobj.bad_pixels,
                                                                                      nodes=nodes,
                                                                                      star_model=dataobj.star_func(dataobj.wavelengths),
                                                                                      threshold=threshold_badpix,
                                                                                      use_set_nans=False,
                                                                                      mypool=mypool,
                                                                                      regularization=True,
                                                                                      reg_mean_map=reg_mean_map0,
                                                                                      reg_std_map=reg_std_map0)

            wherenan = np.where(np.isnan(spline_paras))
            reg_mean_map = copy(spline_paras)
            reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras, axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
            reg_std_map = np.abs(spline_paras)
            reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras), axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
            reg_std_map = reg_std_map
            reg_std_map = np.clip(reg_std_map, 1e-11, np.inf)
            # if 1:
            #     with fits.open(dataobj.starsub_filename) as hdulist:
            #         star_model = hdulist[2].data
            #     data_smooth = star_model/dataobj.star_func(dataobj.wavelengths)
            # else:
            #     #/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/RDI_model_refPSF1/jw01414004001_02101_00001_nrs1_cal.fits
            #     RDImodel_filename = os.path.join("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/RDI_model_refPSF1/",os.path.basename(filename))
            #     hdulist_sc = fits.open(RDImodel_filename)
            #     RDImodel_im = hdulist_sc["SCI"].data
            #     from breads.instruments.jwstnirpsec_cal import normalize_rows
            #
            #     threshold_badpix=10
            #     DImodel_star_model, _, new_badpixs, subtracted_im = normalize_rows(RDImodel_im, dataobj.wavelengths, noise=dataobj.noise, badpixs=dataobj.bad_pixels,
            #                                                                nodes=nodes,
            #                                                                star_model=dataobj.star_func(dataobj.wavelengths),
            #                                                                threshold=threshold_badpix, use_set_nans=False,
            #                                                                mypool=mypool)
            #     DImodel_star_model, _, new_badpixs, subtracted_im = normalize_rows(RDImodel_im, dataobj.wavelengths, noise=dataobj.noise, badpixs=new_badpixs,
            #                                                                nodes=nodes,
            #                                                                star_model=dataobj.star_func(dataobj.wavelengths),
            #                                                                threshold=threshold_badpix, use_set_nans=False,
            #                                                                mypool=mypool)
            #     data_smooth = DImodel_star_model/dataobj.star_func(dataobj.wavelengths)
            #
            #
            # reg_mean_map = np.zeros((dataobj.data.shape[0],nodes))
            # reg_std_map = np.zeros((dataobj.data.shape[0],nodes))
            # for rowid, row in enumerate(data_smooth):
            #     row_wvs = dataobj.wavelengths[rowid,:]
            #     row_bp = dataobj.bad_pixels[rowid,:]
            #     if np.nansum(np.isfinite(row*row_bp)) == 0:
            #         continue
            #     # if rowid < 300:
            #     #     continue
            #     # plt.plot(dataobj.data[rowid,:],label="data")
            #     # plt.plot(row,label="smooth")
            #     row_filled = np.array(pd.DataFrame(row).rolling(window=10, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")).T[0]
            #     wherefiniterow = np.where(np.isfinite(row_bp))
            #     row_f = interp1d(row_wvs[wherefiniterow], row_filled[wherefiniterow], bounds_error=False, fill_value=np.nan)
            #     reg_mean_row = row_f(x_knots)
            #     wherenan = np.where(np.isnan(reg_mean_row))
            #     reg_mean_row = np.array(pd.DataFrame(reg_mean_row).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")).T[0]
            #
            #     reg_std_row = reg_mean_row / 5
            #     reg_std_row[wherenan] = reg_mean_row[wherenan]
            #
            #     reg_mean_map[rowid,:] = reg_mean_row
            #     reg_std_map[rowid,:] = reg_std_row

        from breads.fm.hc_atmgrid_splinefm_jwst_nirspec_cal import hc_atmgrid_splinefm_jwst_nirspec_cal
        fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                    "radius_as":0.1,"badpixfraction":0.75,"nodes":x_knots,"fix_parameters":fix_parameters,
                    "wvs_KLs_f":wvs_KLs_f_list,"regularization":"user","reg_mean_map":reg_mean_map,"reg_std_map":reg_std_map}#,"detec_KLs":detec_KLs
# SNR -4.59363934884126
        fm_func = hc_atmgrid_splinefm_jwst_nirspec_cal

        if 0: # generate fake cal.fits for calibration purposes
            fm_paras_tmp = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                        "radius_as":0.5,"badpixfraction":0.75,"nodes":nodes,"fix_parameters":fix_parameters,"Nrows_max":500}
            nonlin_paras = [0, ra_offset,dec_offset]
            d, M, s,where_finite = fm_func(nonlin_paras,dataobj,return_where_finite=True,**fm_paras_tmp)
            if detector == "nrs2":
                planet2inject = 40e-12*M[:,0]
            elif detector == "nrs1":
                planet2inject = 30e-12*M[:,0]
            hdulist_sc_simu = fits.open(filename)
            simu_photcalib = np.zeros(hdulist_sc_simu["SCI"].data.shape)
            simu_photcalib[np.where(np.isnan(hdulist_sc_simu["SCI"].data))] = np.nan
            simu_photcalib[where_finite] += planet2inject
            hdulist_sc_simu["SCI"].data = simu_photcalib
            if not os.path.exists(os.path.join(utils_dir,"photcalib")):
                os.makedirs(os.path.join(utils_dir,"photcalib"))
            try:
                hdulist_sc_simu.writeto(os.path.join(utils_dir,"photcalib",os.path.basename(filename)), overwrite=True)
            except TypeError:
                hdulist_sc_simu.writeto(os.path.join(utils_dir,"photcalib",os.path.basename(filename)), clobber=True)
            hdulist_sc_simu.close()

        if 0: #inject fake planet in reduction for contrast curve calibration
            fm_paras_tmp = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                            "radius_as":0.5,"badpixfraction":0.99,"nodes":nodes,"fix_parameters":fix_parameters,
                            "Nrows_max":500}
            nonlin_paras = [0, 0.5, -0.5]
            d, M, s,where_finite = fm_func(nonlin_paras,dataobj,return_where_finite=True,**fm_paras_tmp)
            if detector == "nrs1":
                dataobj.data[where_finite] += 2*1.5e-12 * M[:,0]
            elif detector == "nrs2":
                dataobj.data[where_finite] += 2*2e-12 * M[:,0]

        # /!\ Optional but recommended
        # Test the forward model for a fixed value of the non linear parameter.
        # Make sure it does not crash and look the way you want
        if 0:
            # fm_paras["fix_parameters"]= [None,None,None,sc_fib]
            print(ra_offset,dec_offset)
            print(dataobj.wpsf_ra_offset,dataobj.wpsf_dec_offset)
            # nonlin_paras = [0.0,ra_offset+dataobj.wpsf_ra_offset,dec_offset+dataobj.wpsf_dec_offset] # x (pix),y (pix), rv (km/s)
            nonlin_paras = [0, -1.044, -0.469] # x (pix),y (pix), rv (km/s)
            # nonlin_paras = [0, -1.4, -1.4] # x (pix),y (pix), rv (km/s)
            # d is the data vector a the specified location
            # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
            # s is the vector of uncertainties corresponding to d
            d, M, s,extra_outputs = fm_func(nonlin_paras,dataobj,return_where_finite=True,**fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            # s = np.ones(s.shape)
            print(np.unique(where_finite[0]))
            # exit()
            mask_comp = np.zeros(dataobj.data.shape)
            mask_comp[where_finite] = 1
            with fits.open(dataobj.starsub_filename) as hdulist:
                subtracted_im = hdulist[0].data
            # plt.subplot(1,2,1)
            # plt.imshow(dataobj.wavelengths,origin="lower")
            # plt.subplot(1,2,2)
            # plt.imshow(subtracted_im,origin="lower")
            # contour = plt.gca().contour(mask_comp.astype(int), colors='cyan', levels=[0.5])
            # plt.clim([-1e-11,1e-11])
            # plt.show()

            # d,M,s = d[:1717],M[:1717,:],s[0:1717]
            # d,M,s = d[500:1000],M[500:1000,:],s[500:1000]

            wvs_im = dataobj.wavelengths
            cols_im = np.tile(np.arange(wvs_im.shape[1])[None,:],(wvs_im.shape[0],1))
            d_wvs = wvs_im[where_finite]
            d_cols = cols_im[where_finite]

            print(M.shape)
            # validpara = np.where(np.sum(M,axis=0)!=0)
            validpara = np.where(np.max(np.abs(M),axis=0)!=0)
            M = M[:,validpara[0]]
            # print(M.shape)
            # print(d.shape)
            # print(np.nanmax(M,axis=0))
            # print(np.sum(M,axis=0)!=0)
            # print(np.nanstd(M,axis=0))

            d = d / s
            M = M / s[:, None]
            from scipy.optimize import lsq_linear

            # plt.subplot(3,1,1)
            # plt.plot(d,label="data")
            # plt.subplot(3,1,2)
            # plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
            # # plt.legend()
            # plt.subplot(3,1,3)
            # for k in range(M.shape[-1]-1):
            #     plt.plot(M[:,k+1])
            # plt.legend()
            # plt.show()

            from breads.fit import fitfm
            log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0 = True,bounds = None,residuals=None,residuals_H0=None)
            # print("here")
            # linparas0 = np.zeros(M_full.shape[1])+np.nan
            # linparas0[validpara] = paras
            # print(linparas0)
            # print("fitfm")
            # print(linparas)
            paras = linparas[validpara]
            print("best fit", linparas[0:5])
            print("best fit err",linparas_err[0:5])
            print("best fit snr",linparas[0:5]/linparas_err[0:5])#[107.89260376   4.30915356   4.32027028   4.39081023   9.39263101]
            # paras = lsq_linear(M, d).x
            # paras_H0 = lsq_linear(M[:,1::], d).x
            # m_H0 = np.dot(M[:,1::],paras_H0)
            # print("best fit",paras[0:5])
            # print((paras[0]*u.MJy *(const.c)/(photfilter_wv0*u.um)**2).to( u.W/u.m**2/u.um))
            # print((paras[0]*u.MJy *(const.c)/(photfilter_wv0*u.um)).to( u.W/u.m**2))

            logdet_Sigma = np.sum(2 * np.log(s))
            m = np.dot(M, paras)
            r = d  - m
            chi2 = np.nansum(r**2)
            N_data = np.size(d)
            rchi2 = chi2 / N_data
            # MTM = np.dot(M.T, M)
            # covphi = rchi2 * np.linalg.inv(MTM)
            # slogdet_icovphi0 = np.linalg.slogdet(MTM)

            # N_linpara = M.shape[1]
            # from scipy.special import loggamma
            # log_prob = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0[1] - (N_data - N_linpara + 2 - 1) / 2 * np.log(chi2) + \
            #             loggamma((N_data - N_linpara + 2 - 1) / 2) + (N_linpara - N_data) / 2 * np.log(2 * np.pi)
            # diagcovphi = copy(np.diag(covphi))
            # diagcovphi[np.where(diagcovphi<0.0)] = np.nan
            # paras_err = np.sqrt(diagcovphi)
            # print("best fit err",paras_err[0:5])
            # print("SNR",paras[0]/paras_err[0])

            #
            # Fl_spec = ((M[:,0]*s*u.MJy)*const.c/(photfilter_wv0*u.um)**2).to( u.W/u.m**2/u.um)
            # w= dataobj.wavelengths[where_finite]
            # dw= dataobj.delta_wavelengths[where_finite]
            # filter_norm = np.nansum((dw*u.um)*photfilter_f(w))
            # Flambda = np.nansum((dw*u.um)*photfilter_f(w)*Fl_spec)/filter_norm # u.W*u.m**-2/u.um
            # Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
            # print((Flambda*photfilter_wv0*u.um).to(u.W*u.m**-2),Fnu.to(u.MJy))
            # print(paras[0]*(Flambda*photfilter_wv0*u.um).to(u.W*u.m**-2),paras[0]*Fnu.to(u.MJy))
            # exit()

            res = r * s

            # log_prob,log_prob_H0,rchi2,linparas,linparas_err= grid_search([[nonlin_paras[0]], [nonlin_paras[1]], [nonlin_paras[2]]], dataobj, fm_func, fm_paras, numthreads=None)
            # print(linparas[0,0,0,:])
            # print(linparas_err[0,0,0,:])

            plt.subplot(3,1,1)
            plt.plot(d*s,label="data")
            plt.plot(m*s,label="Combined model")
            plt.plot(paras[0]*M[:,0]*s,label="planet model")
            plt.plot((m-paras[0]*M[:,0])*s,label="starlight model")
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()
            plt.subplot(3,1,2)
            plt.plot(r,label="Residuals")
            plt.fill_between(np.arange(np.size(s)),-1,1,label="Error",alpha=0.5)
            r_std = np.nanstd(r)
            plt.ylim([-10*r_std,10*r_std])
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()
            plt.subplot(3,1,3)
            plt.plot(M[:,0],label="planet model")
            plt.subplot(4,1,4)
            for k in range(np.min([100,M.shape[-1]-1])):
                plt.plot(M[:,k+1],label="starlight model {0}".format(k+1))
            plt.legend()
            # plt.figure(4)
            # # plt.subplot(2,1,1)
            # cumsum_res = np.cumsum((d-m)**2)
            # # print("coucou",np.polyfit(np.arange(np.size(cumsum_res)),cumsum_res,1))
            # cumsum_res -= np.polyval(np.polyfit(np.arange(np.size(cumsum_res)),cumsum_res,1),np.arange(np.size(cumsum_res)))
            # plt.plot(cumsum_res,label="H1")
            # # plt.plot(np.cumsum((d-m_H0)**2),label="H0")
            # # plt.subplot(2,1,2)
            # # plt.plot(np.cumsum((d-m_H0)**2)-np.cumsum((d-m)**2),label="H0-H1")
            # # plt.subplot(3,1,3)
            # # plt.scatter(d_wvs,np.cumsum((d-m_H0)**2)-np.cumsum((d-m)**2),label="H0-H1")
            # # plt.tight_layout()

            plt.figure(2)
            plt.subplot(3,1,1)
            plt.scatter(d_wvs,res,alpha=0.5,s=0.1)
            plt.xlabel("Wavelength (um)")
            plt.subplot(3,1,2)
            plt.scatter(d_cols,res,alpha=0.5,s=0.1)
            plt.xlabel("Column index")
            plt.tight_layout()
            plt.subplot(3,1,3)
            fcal= "/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/stage2_notelegraph/jw01414004001_02101_00003_nrs1_cal.fits"
            hdulist_sc = fits.open(fcal)
            im = hdulist_sc["SCI"].data
            # plt.plot(im[220,:])
            random_ints = np.array([219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 268, 269])
            for row in random_ints:#random_ints:
                plt.plot(im[row,:],alpha=0.5)
            plt.ylim([-1e-10,1e-10])

            # plt.figure(3)
            # plt.scatter(d_wvs,d_cols,alpha=0.5,s=0.1)
            # plt.xlabel("Wavelength (um)")
            # plt.ylabel("Column index")
            plt.show()


        if 1:
            rvs = np.array([0])
            # rvs = np.linspace(-4000,4000,21,endpoint=True)
            # ra_offset,dec_offset = 0.5,-0.5
            # ras = np.arange(ra_offset-0.0,ra_offset+0.6,0.1)
            # decs = np.arange(dec_offset-0.0,dec_offset+0.6,0.1)
            # ras = np.arange(-2.5,2.5,0.1)
            # decs = np.arange(-3.0,2.0,0.1)
            ras = np.arange(-2.5,2.5,0.05)
            decs = np.arange(-3.0,2.0,0.05)
            # ras = np.arange(-2.0,-1.5,0.1)
            # decs = np.array([0])
            if 0:
                import cProfile
                import pstats
                cProfile.run("grid_search([rvs,ras,decs],dataobj,fm_func,fm_paras,numthreads=None)",
                             '../profiling_results')
                # Load the profiling results into a pstats object
                profiler = pstats.Stats('profiling_results')
                # Sort the results by tottime (total time spent in a function)
                profiler.sort_stats('cumtime')
                # Print the sorted profiling results
                profiler.print_stats()
                exit()
            # log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,ras,decs],dataobj,fm_func,fm_paras,numthreads=None)
            log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,ras,decs],dataobj,fm_func,fm_paras,numthreads=numthreads)
            N_linpara = linparas.shape[-1]

            import datetime
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y%m%d_%H%M%S")


            outoftheoven_filename = os.path.join(out_dir,formatted_datetime+"_"+os.path.basename(filename).replace(".fits","_out.fits"))
            print(outoftheoven_filename)
            with h5py.File(outoftheoven_filename, 'w') as hf:
                hf.create_dataset("rvs", data=rvs)
                hf.create_dataset("ras", data=ras)
                hf.create_dataset("decs", data=decs)
                hf.create_dataset("log_prob", data=log_prob)
                hf.create_dataset("log_prob_H0", data=log_prob_H0)
                hf.create_dataset("rchi2", data=rchi2)
                hf.create_dataset("linparas", data=linparas)
                hf.create_dataset("linparas_err", data=linparas_err)
        else:
            print(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
            outoftheoven_filelist = glob(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
            outoftheoven_filelist.sort()
            outoftheoven_filename = outoftheoven_filelist[-1]
            print(outoftheoven_filename)
            # outoftheoven_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/out/20230417_231940_jw01414004001_02101_00001_nrs2_cal_out.fits"
            with h5py.File(outoftheoven_filename, 'r') as hf:
                rvs = np.array(hf.get("rvs"))
                ras = np.array(hf.get("ras"))
                decs = np.array(hf.get("decs"))
                log_prob = np.array(hf.get("log_prob"))
                log_prob_H0 = np.array(hf.get("log_prob_H0"))
                rchi2 = np.array(hf.get("rchi2"))
                linparas = np.array(hf.get("linparas"))
                linparas_err = np.array(hf.get("linparas_err"))

    mypool.close()
    mypool.join()

    k,l,m = np.unravel_index(np.nanargmax(log_prob-log_prob_H0),log_prob.shape)
    print("best fit parameters: rv={0},y={1},x={2}".format(rvs[k],ras[l],decs[m]) )
    print(np.nanmax(log_prob-log_prob_H0))
    best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[rvs[k]], [ras[l]], [decs[m]]], dataobj, fm_func, fm_paras, numthreads=None)
    print(best_log_prob-best_log_prob_H0)

    linparas = np.swapaxes(linparas, 1, 2)
    linparas_err = np.swapaxes(linparas_err, 1, 2)
    log_prob = np.swapaxes(log_prob, 1, 2)
    log_prob_H0 = np.swapaxes(log_prob_H0, 1, 2)
    rchi2 = np.swapaxes(rchi2, 1, 2)

    dra=ras[1]-ras[0]
    ddec=decs[1]-decs[0]
    ras_grid,decs_grid = np.meshgrid(ras,decs)
    rs_grid = np.sqrt(ras_grid**2+decs_grid**2)
    print(ra_offset,dec_offset)
    rs_comp_grid = np.sqrt((ras_grid-ra_offset)**2+(decs_grid-dec_offset)**2)
    # plt.imshow(rs_comp_grid,origin="lower")
    # plt.show()

    plt.figure(1)
    plt.subplot(1,3,1)
    plt.title("SNR map")
    snr_map = linparas[k,:,:,0]/linparas_err[k,:,:,0]
    # print("SNR std",np.nanmax(snr_map[np.where(rs_comp_grid>0.4)]))
    print("SNR std",np.nanstd(snr_map[np.where((rs_comp_grid>0.7)*np.isfinite(snr_map))]))
    plt.imshow(snr_map,origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    plt.clim([-2,5])
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    # plt.plot(out[:,0,0,2])
    plt.xlabel("dRA (as)")
    plt.ylabel("ddec (as)")

    contrast_5sig  = 5*linparas_err[k,:,:,0]/HD19467_flux_MJy[photfilter_name]
    print(HD19467_flux_MJy[photfilter_name])
    nan_mask_boxsize=2
    contrast_5sig[np.where(np.isnan(correlate2d(contrast_5sig,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan

    plt.subplot(1,3,2)
    plt.title("Flux map")
    plt.imshow(linparas[k,:,:,0],origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    cbar = plt.colorbar()
    cbar.set_label("planet flux MJy")
    plt.xlabel("dRA (as)")
    plt.ylabel("ddec (as)")
    plt.subplot(1,3,3)
    plt.title("5-$\sigma$ Sensitivity 2D {0}".format(photfilter_name))
    plt.imshow(np.log10(contrast_5sig),origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    # plt.clim([0,100])
    plt.xlabel("dRA (as)")
    plt.ylabel("ddec (as)")
    cbar = plt.colorbar()
    cbar.set_label("5-$\sigma$ Flux ratio log10 ({0})".format(photfilter_name))

    plt.figure(3)
    plt.title("5-$\sigma$ Sensitivity 1D")
    plt.scatter(rs_grid,contrast_5sig)
    plt.yscale("log")
    plt.xlabel("Separation (as)")
    plt.ylabel("5-$\sigma$ Flux ratio ({0})".format(photfilter_name))

    plt.figure(4)
    snr_map_masked = copy(snr_map)
    snr_map_masked[np.where((rs_comp_grid < 0.7))] = np.nan

    # Create a histogram using the hist function from NumPy
    hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20*3, range=(-10, 10))#, bins=256, range=(0, 256)
    bin_centers = (bins[1::]+bins[0:np.size(bins)-1])/2.
    plt.plot(bin_centers,hist/(np.nansum(hist)*(bins[1]-bins[0])),label="snr map")
    plt.plot(bin_centers,1/np.sqrt(2*np.pi)*np.exp(-0.5*(bin_centers-0.0)**2),color="black",linestyle="--",label="Gaussian")
    plt.yscale("log")
    plt.ylim([1e-4,1])
    plt.legend()
    plt.show()




    exit()