import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
import h5py
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const
from scipy.signal import correlate2d

import matplotlib.patheffects as PathEffects
from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar

def fill_gap(HD19467B_wvs,HD19467B_spec,planet_f,fit_range= [3.9,4.3]):


    HD19467B_wvs = HD19467B_wvs[np.where(np.isfinite(HD19467B_spec))]
    HD19467B_spec = HD19467B_spec[np.where(np.isfinite(HD19467B_spec))]
    HD19467B_dwvs = HD19467B_wvs[1::]-HD19467B_wvs[0:np.size(HD19467B_wvs)-1]
    HD19467B_dwvs = np.insert(HD19467B_dwvs,0,HD19467B_dwvs[0])
    startid_2ndhalf = np.where(HD19467B_dwvs>0.05)[0][0]
    dw = np.nanmedian(HD19467B_dwvs)
    gap_wvs = np.arange(HD19467B_wvs[startid_2ndhalf-1]+dw,HD19467B_wvs[startid_2ndhalf],dw)
    HD19467B_gapfilled_wvs = np.concatenate([HD19467B_wvs[0:startid_2ndhalf],gap_wvs,HD19467B_wvs[startid_2ndhalf::]])

    rv=0
    where_wvs4fit = np.where((HD19467B_wvs>fit_range[0])*(HD19467B_wvs<fit_range[1]))
    HD19467B_wvs_4fit = HD19467B_wvs[where_wvs4fit]
    HD19467B_spec_4fit = HD19467B_spec[where_wvs4fit]
    comp_spec_4fit = planet_f(HD19467B_wvs_4fit * (1 - (rv) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    comp_spec_4fit = comp_spec_4fit * (HD19467B_wvs_4fit * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec_4fit = comp_spec_4fit.to(u.MJy).value

    comp_spec_scale = np.nansum(HD19467B_spec_4fit*comp_spec_4fit)/np.nansum(comp_spec_4fit**2)

    comp_spec_gap = planet_f(gap_wvs * (1 - (rv) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    comp_spec_gap = comp_spec_gap * (gap_wvs * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec_gap = comp_spec_scale * comp_spec_gap.to(u.MJy).value
    HD19467B_gapfilled_spec = np.concatenate([HD19467B_spec[0:startid_2ndhalf],comp_spec_gap,HD19467B_spec[startid_2ndhalf::]])

    return HD19467B_gapfilled_wvs,HD19467B_gapfilled_spec

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    ra_offset = -1.33  # ra offset in as
    dec_offset = -0.876  # dec offset in as
    ra_corr,dec_corr = -0.1285876002234175, - 0.06997868615326872
    print(ra_offset+ra_corr,dec_offset+dec_corr)

    color_list = ["#ff9900","#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"

    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}
    HD19467B_Jens_flux_MJy = {"F250M":18.4e-12, # in MJy, Jens Kammerer
                         "F300M":36e-12,
                         "F360M":54.3e-12,
                         "F410M":166.1e-12,
                         "F430M":111.6e-12,
                         "F460M":76.5e-12}
    HD19467B_Jens_fluxerr_MJy = {"F250M":0.9e-12, # in MJy, Jens Kammerer
                             "F300M":0.5e-12,
                             "F360M":0.6e-12,
                             "F410M":1.4e-12,
                             "F430M":1.3e-12,
                             "F460M":1.6e-12}

    F444W_Carter = np.loadtxt("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/HIP65426_ERS_CONTRASTS/F444W_ADI+RDI.txt")
    F444W_Carter_seps = F444W_Carter[:, 0]
    F444W_Carter_cons = F444W_Carter[:, 1]
    # plt.plot(F444W_Carter_seps,F444W_Carter_cons)
    # plt.yscale("log")
    # plt.show()

    #Download NIRCam filters from http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=JWST&gname2=NIRCam&asttype=
    # Select Data file: ascii
    for photfilter_name in HD19467B_Jens_flux_MJy.keys():
        photfilter = os.path.join(external_dir,"JWST_NIRCam."+photfilter_name+".dat")
        filter_arr = np.loadtxt(photfilter)
        trans_wvs = filter_arr[:, 0] / 1e4
        trans = filter_arr[:, 1]
        photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
        photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
        bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
        photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]
        print(photfilter_name, photfilter_wvmin, photfilter_wvmax)

    if 1:
        # wvs_phoenix = "/scr3/jruffio/data/kpic/models/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
        # star_phoenix = "/scr3/jruffio/models/phoenix/fitting/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        # with pyfits.open(wvs_phoenix) as hdulist:
        #     phoenix_wvs = hdulist[0].data / 1.e4
        # crop_phoenix = np.where((phoenix_wvs > minwv - 0.02) * (phoenix_wvs < maxwv + 0.02))
        # phoenix_wvs = phoenix_wvs[crop_phoenix]
        # with pyfits.open(A0_phoenix) as hdulist:
        #     phoenix_A0 = hdulist[0].data[crop_phoenix]
        # print("broadening A0 phoenix model")
        # phoenix_A0_broad = dataobj.broaden(phoenix_wvs, phoenix_A0, loc=sc_fib, mppool=mypool)
        # phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_broad, bounds_error=False, fill_value=np.nan)
        # phoenix_wvs_reg = np.arange(np.min(phoenix_wvs),np.max(phoenix_wvs),3e-6)
        # phoenix_A0_func(phoenix_wvs_reg)
        # spinbroad_phoenix_A0 = pyasl.fastRotBroad(phoenix_wvs_reg, phoenix_A0_func(phoenix_wvs_reg), 0.1, vsiniA0)
        # phoenix_A0_func = interp1d(phoenix_wvs_reg, spinbroad_phoenix_A0, bounds_error=False, fill_value=np.nan)

        photfilter_name_nrs1,wv_sampling_nrs1 = "F360M", np.arange(2.859509, 4.1012874, 0.0006763935)
        photfilter_name_nrs2,wv_sampling_nrs2 = "F460M",np.arange(4.081285,5.278689,0.0006656647)#"F430M"#"F460M"#"F410M"#
        HD19467A_wvs = np.concatenate([wv_sampling_nrs1,wv_sampling_nrs2])
        fitpsf_filename_nrs1 = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/jw01414004001_02101_nrs1_fitpsf_webbpsf.fits"
        fitpsf_filename_nrs2 = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/jw01414004001_02101_nrs2_fitpsf_webbpsf.fits"
        with fits.open(fitpsf_filename_nrs1) as hdulist:
            bestfit_coords1 = hdulist[0].data
        with fits.open(fitpsf_filename_nrs2) as hdulist:
            bestfit_coords2 = hdulist[0].data
        HD19467A_spec = np.concatenate([bestfit_coords1[:,0],bestfit_coords2[:,0]])
        where_good = np.where(np.isfinite(HD19467A_spec)*((HD19467A_wvs<4.02)+(HD19467A_wvs>4.22)))
        HD19467A_wvs = HD19467A_wvs[where_good]
        HD19467A_spec = HD19467A_spec[where_good]
        z = np.polyfit(HD19467A_wvs[np.where(np.isfinite(HD19467A_spec))], HD19467A_spec[np.where(np.isfinite(HD19467A_spec))], 2)
        print(z)
        star_f = interp1d(np.arange(2.5,5.5,0.01), np.polyval(z,np.arange(2.5,5.5,0.01)), bounds_error=False, fill_value=0)

        HD19467A_dwvs = HD19467A_wvs[1::] - HD19467A_wvs[0:np.size(HD19467A_wvs) - 1]
        HD19467A_dwvs = np.insert(HD19467A_dwvs, 0, HD19467A_dwvs[0])
        startid_2ndhalf = np.where(HD19467A_dwvs > 0.05)[0][0]
        dw = np.nanmedian(HD19467A_dwvs)
        gap_wvs = np.arange(HD19467A_wvs[startid_2ndhalf - 1] + dw, HD19467A_wvs[startid_2ndhalf], dw)
        comp_spec_gap = np.polyval(z, gap_wvs)
        HD19467A_gapfilled_wvs = np.concatenate([HD19467A_wvs[0:startid_2ndhalf], gap_wvs, HD19467A_wvs[startid_2ndhalf::]])
        HD19467A_gapfilled_spec = np.concatenate([HD19467A_spec[0:startid_2ndhalf],comp_spec_gap,HD19467A_spec[startid_2ndhalf::]])
        HD19467A_gapfilled_dwvs = HD19467A_gapfilled_wvs[1::]-HD19467A_gapfilled_wvs[0:np.size(HD19467A_gapfilled_wvs)-1]
        HD19467A_gapfilled_dwvs = np.insert(HD19467A_gapfilled_dwvs,0,HD19467A_gapfilled_dwvs[0])

        # plt.plot(HD19467A_wvs[np.where(np.isfinite(HD19467A_spec))], HD19467A_spec[np.where(np.isfinite(HD19467A_spec))])
        # plt.plot(HD19467A_gapfilled_wvs,HD19467A_gapfilled_spec,linestyle="--")
        # plt.plot(np.arange(2.5,5.5,0.01), np.polyval(z,np.arange(2.5,5.5,0.01)),linestyle=":")
        # plt.show()

        HD19467A_NIRSpec_flux_MJy = {}
        for photfilter_name in ["F360M", "F410M", "F430M", "F460M", "F444W", "F356W"]:  # HD19467_flux_MJy.keys():
            photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/JWST_NIRCam." + photfilter_name + ".dat"
            filter_arr = np.loadtxt(photfilter)
            trans_wvs = filter_arr[:, 0] / 1e4
            trans = filter_arr[:, 1]
            photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
            photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
            bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
            photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]

            filter_norm = np.nansum((HD19467A_gapfilled_dwvs * u.um) * photfilter_f(HD19467A_gapfilled_wvs))
            aper_spec_Flambda = (HD19467A_gapfilled_spec * u.MJy * const.c / (HD19467A_gapfilled_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um)
            Flambda = np.nansum((HD19467A_gapfilled_dwvs * u.um) * photfilter_f(HD19467A_gapfilled_wvs) * (aper_spec_Flambda)) / filter_norm
            Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
            Fnu_Mjy = Fnu.to(u.MJy).value
            HD19467A_NIRSpec_flux_MJy[photfilter_name] = Fnu_Mjy

        HD19467_flux_MJy["F444W"] = HD19467_flux_MJy["F460M"]/HD19467A_NIRSpec_flux_MJy["F460M"]*HD19467A_NIRSpec_flux_MJy["F444W"]
        HD19467_flux_MJy["F356W"] = HD19467_flux_MJy["F360M"]/HD19467A_NIRSpec_flux_MJy["F360M"]*HD19467A_NIRSpec_flux_MJy["F356W"]
    # print(HD19467_flux_MJy["F444W"] )
    # exit()

    mast_s3d = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/jw01414-o004_t003_nirspec_g395h-f290lp_s3d.fits"
    hdulist_sc = fits.open(mast_s3d)
    cube = hdulist_sc["SCI"].data
    HD19467_median_im = np.nanmedian(cube,axis=0)
    # plt.imshow(np.log10(HD19467_median_im),origin="lower")
    # # plt.clim([0,3e-8])
    # plt.clim([-14,-7])
    # plt.show()
    kc,lc = 32,23# np.unravel_index(np.nanargmax(HD19467_median_im),HD19467_median_im.shape)
    s3d_Dec_vec = (np.arange(HD19467_median_im.shape[0])-kc)*0.1 #- ra_corr
    s3d_RA_vec = (np.arange(HD19467_median_im.shape[1])-lc)*0.1 #- dec_corr
    s3d_dDec = s3d_Dec_vec[1]-s3d_Dec_vec[0]
    s3d_dRA = s3d_RA_vec[1]-s3d_RA_vec[0]

    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"
    with h5py.File(os.path.join(external_dir,"BT-Settl15_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"), 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs = np.array(hf.get("wvs"))
    grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
    grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
    filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
    Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs)[None, None, :] * (
                grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
    Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
    grid_specs = grid_specs / Fnu[:, :, None].to(u.MJy).value
    myinterpgrid = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear",
                                           bounds_error=False, fill_value=np.nan)
    planet_f = interp1d(grid_wvs, myinterpgrid((1000, 5.0)), bounds_error=False, fill_value=0)

    denominator = 0
    seps_5sig_Jens = np.arange(0,3,0.1)
    for photfilter_name in HD19467B_Jens_flux_MJy.keys():
        # if photfilter_name =="F250M":
        #     continue
        if photfilter_name !="F410M":
            continue
            #/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/ADI_NANNU1_NSUBS1_JWST_NIRCAM_NRCALONG_F460M_MASKBAR_MASKALWB_SUB320ALWB-KLmodes-all_seps.npy
        seps_Jens = np.load("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/ADI_NANNU1_NSUBS1_JWST_NIRCAM_NRCALONG_"+photfilter_name+"_MASKBAR_MASKALWB_SUB320ALWB-KLmodes-all_seps.npy")
        cons_Jens = np.load("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/ADI_NANNU1_NSUBS1_JWST_NIRCAM_NRCALONG_"+photfilter_name+"_MASKBAR_MASKALWB_SUB320ALWB-KLmodes-all_cons.npy")
        cons_Jens_f = interp1d(seps_Jens[0],np.nanmin(cons_Jens,axis=0),bounds_error=False,fill_value=np.nan)
        # plt.plot(seps_Jens[0],np.nanmin(cons_Jens,axis=0)*(HD19467B_Jens_flux_MJy["F460M"]/HD19467B_Jens_flux_MJy[photfilter_name])*(HD19467_flux_MJy[photfilter_name]/HD19467_flux_MJy["F460M"]),label=photfilter_name)
        denominator = denominator+1/((HD19467_flux_MJy[photfilter_name]*cons_Jens_f(seps_5sig_Jens)/5.)*(HD19467B_Jens_flux_MJy["F460M"]/HD19467B_Jens_flux_MJy[photfilter_name]))**2
    cont_5sig_Jens = 5*np.sqrt(1/denominator)/HD19467_flux_MJy["F460M"]
    # plt.plot(seps_5sig_Jens,cont_5sig_Jens,label="combined")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    A0_filename = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/MAST_2023-04-23T0044/JWST/jw01128-o009_t007_nirspec_g395h-f290lp/jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits"
    hdulist_sc = fits.open(A0_filename)
    cube = hdulist_sc["SCI"].data
    im_psfprofile = np.nanmedian(cube,axis=0)
    im_psfprofile /= np.nanmax(im_psfprofile)
    # kc,lc = np.unravel_index(np.nanargmax(im_psfprofile),im_psfprofile.shape)
    kc, lc = 38.5,34.5
    kvec = (np.arange(im_psfprofile.shape[0])-kc)*0.1
    lvec = (np.arange(im_psfprofile.shape[1])-lc)*0.1
    lgrid,kgrid = np.meshgrid(lvec,kvec)
    rgrid_psfprofile = np.sqrt(lgrid**2+kgrid**2)
    im_psfprofile[np.where(rgrid_psfprofile>1.5)] = np.nan
    # plt.imshow(im_psfprofile,origin="lower")
    # plt.show()



    RDI_1dspectrum_speckles_filename = os.path.join(out_png, "HD19467b_RDI_1dspectrum_speckles.fits")
    with fits.open(RDI_1dspectrum_speckles_filename) as hdulist:
        HD19467B_speckles_wvs = hdulist[0].data
        HD19467B_speckles_spec = hdulist[1].data

    RDI_1dspectrum_filename = os.path.join(out_png, "HD19467b_RDI_1dspectrum.fits")
    with fits.open(RDI_1dspectrum_filename) as hdulist:
        HD19467B_wvs = hdulist[0].data
        HD19467B_spec = hdulist[1].data
        HD19467B_spec_err = hdulist[2].data
    HD19467B_dwvs = HD19467B_wvs[1::]-HD19467B_wvs[0:np.size(HD19467B_wvs)-1]
    HD19467B_dwvs = np.insert(HD19467B_dwvs,0,HD19467B_dwvs[0])

    if 1: #fill gap
        HD19467B_gapfilled_wvs,HD19467B_gapfilled_spec = fill_gap(HD19467B_wvs,HD19467B_spec,planet_f)
        HD19467B_gapfilled_dwvs = HD19467B_gapfilled_wvs[1::]-HD19467B_gapfilled_wvs[0:np.size(HD19467B_gapfilled_wvs)-1]
        HD19467B_gapfilled_dwvs = np.insert(HD19467B_gapfilled_dwvs,0,HD19467B_gapfilled_dwvs[0])
        #1785, 1786
        # plt.plot(HD19467B_gapfilled_wvs,HD19467B_gapfilled_spec)
        # # plt.plot(gap_wvs,comp_spec_gap)
        # plt.show()

    HD19467B_NIRSpec_flux_MJy = {}
    HD19467B_NIRSpec_fluxerr_MJy = {}
    for photfilter_name in ["F360M","F410M","F430M","F460M","F444W","F356W"]:#HD19467_flux_MJy.keys():
        photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/JWST_NIRCam." + photfilter_name + ".dat"
        filter_arr = np.loadtxt(photfilter)
        trans_wvs = filter_arr[:, 0] / 1e4
        trans = filter_arr[:, 1]
        photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
        photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
        bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
        photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]

        filter_norm = np.nansum((HD19467B_gapfilled_dwvs * u.um) * photfilter_f(HD19467B_gapfilled_wvs))

        aper_spec_Flambda = (HD19467B_gapfilled_spec * u.MJy * const.c / (HD19467B_gapfilled_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um)
        Flambda = np.nansum(
            (HD19467B_gapfilled_dwvs * u.um) * photfilter_f(HD19467B_gapfilled_wvs) * (aper_spec_Flambda)) / filter_norm
        Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
        Fnu_Mjy = Fnu.to(u.MJy).value

        Fnu_MJy_speckles = []
        for speckle_spec in HD19467B_speckles_spec.T:
            speckle_gapfilled_wvs,speckle_gapfilled_spec = fill_gap(HD19467B_wvs,speckle_spec,planet_f)
            speckle_gapfilled_dwvs = speckle_gapfilled_wvs[1::]-speckle_gapfilled_wvs[0:np.size(speckle_gapfilled_wvs)-1]
            speckle_gapfilled_dwvs = np.insert(speckle_gapfilled_dwvs,0,speckle_gapfilled_dwvs[0])

            filter_norm = np.nansum((speckle_gapfilled_dwvs * u.um) * photfilter_f(speckle_gapfilled_wvs))
            aper_spec_Flambda = (speckle_gapfilled_spec * u.MJy * const.c / (speckle_gapfilled_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um)
            Flambda = np.nansum(
                (speckle_gapfilled_dwvs * u.um) * photfilter_f(speckle_gapfilled_wvs) * (aper_spec_Flambda)) / filter_norm
            Fnu_speckle = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
            Fnu_MJy_speckles.append(Fnu_speckle.to(u.MJy).value)
        Fnu_err_Mjy = np.nanstd(Fnu_MJy_speckles)

        print(photfilter_name, "{0}+-{1} MJy ({2:.1f}-{3:.1f}um)".format(Fnu_Mjy,Fnu_err_Mjy,photfilter_wvmin, photfilter_wvmax))
        HD19467B_NIRSpec_flux_MJy[photfilter_name] = Fnu_Mjy
        HD19467B_NIRSpec_fluxerr_MJy[photfilter_name] = Fnu_err_Mjy

    # exit()

    F460M_to_F444W = HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467B_NIRSpec_flux_MJy["F460M"]
    nrs2_flux_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_flux.fits")
    with fits.open(nrs2_flux_filename) as hdulist:
        nrs2_flux = hdulist[0].data*F460M_to_F444W
    nrs2_fluxerr_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_fluxerr.fits")
    with fits.open(nrs2_fluxerr_filename) as hdulist:
        nrs2_fluxerr = hdulist[0].data*F460M_to_F444W
    nrs2_RADec_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_RADec.fits")
    with fits.open(nrs2_RADec_filename) as hdulist:
        nrs2_RA_grid = hdulist[0].data - ra_corr
        nrs2_Dec_grid = hdulist[1].data - dec_corr
        nrs2_RA_vec =   nrs2_RA_grid[0,:]
        nrs2_Dec_vec =   nrs2_Dec_grid[:,0]
    dra = nrs2_RA_vec[1]-nrs2_RA_vec[0]
    ddec = nrs2_Dec_vec[1]-nrs2_Dec_vec[0]
    rs_grid = np.sqrt(nrs2_RA_grid**2+nrs2_Dec_grid**2)
    PAs_grid = np.arctan2(nrs2_RA_grid,nrs2_Dec_grid) % (2 * np.pi)
    nrs2_hist_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_hist.fits")
    with fits.open(nrs2_hist_filename) as hdulist:
        nrs2_bins = hdulist[0].data
        nrs2_hist = hdulist[1].data

    F360M_to_F444W = HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467B_NIRSpec_flux_MJy["F360M"]
    nrs1_flux_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_flux.fits")
    with fits.open(nrs1_flux_filename) as hdulist:
        nrs1_flux = hdulist[0].data*F360M_to_F444W
    nrs1_fluxerr_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_fluxerr.fits")
    with fits.open(nrs1_fluxerr_filename) as hdulist:
        nrs1_fluxerr = hdulist[0].data*F360M_to_F444W
    nrs1_RADec_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_RADec.fits")
    with fits.open(nrs1_RADec_filename) as hdulist:
        nrs1_RA_grid = hdulist[0].data - ra_corr
        nrs1_Dec_grid = hdulist[1].data - dec_corr
        nrs1_RA_vec =   nrs1_RA_grid[0,:]
        nrs1_Dec_vec =   nrs1_Dec_grid[:,0]
    nrs1_hist_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_hist.fits")
    with fits.open(nrs1_hist_filename) as hdulist:
        nrs1_bins = hdulist[0].data
        nrs1_hist = hdulist[1].data

    deno = (1/nrs1_fluxerr**2 + 1/nrs2_fluxerr**2)
    combined_flux = (nrs1_flux/nrs1_fluxerr**2 + nrs2_flux/nrs2_fluxerr**2)/deno
    combined_flux_err = 1/np.sqrt(deno)
    combined_SNR = combined_flux/combined_flux_err
    contrast_5sig_combined  = 5*combined_flux_err/HD19467_flux_MJy["F444W"]
    contrast_5sig_combined[np.where(~np.isfinite(contrast_5sig_combined))] = np.nan


    mask= (-0.6+1./6. * np.pi+PAs_grid)% (2./6. * np.pi)
    mask = np.abs(mask-1./6. * np.pi)<(np.pi/24)
    # plt.figure(20)
    # plt.subplot(1,3,1)
    # plt.imshow((PAs_grid)% (2./6. * np.pi),origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
    # plt.xlim([-2.5,2.5])
    # plt.xticks([-2,-1,0,1,2])
    # plt.ylim([-3.0,2.])
    # plt.yticks([-3,-2,-1,0,1,2])
    # plt.gca().set_aspect('equal')
    # plt.gca().invert_xaxis()
    # plt.subplot(1,3,2)
    # plt.imshow(mask,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
    # plt.xlim([-2.5,2.5])
    # plt.xticks([-2,-1,0,1,2])
    # plt.ylim([-3.0,2.])
    # plt.yticks([-3,-2,-1,0,1,2])
    # plt.gca().set_aspect('equal')
    # plt.gca().invert_xaxis()
    # plt.subplot(1,3,3)
    # plt.imshow(np.log10(combined_flux_err/1e-12),origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
    # plt.xlim([-2.5,2.5])
    # plt.xticks([-2,-1,0,1,2])
    # plt.ylim([-3.0,2.])
    # plt.yticks([-3,-2,-1,0,1,2])
    # plt.gca().set_aspect('equal')
    # plt.gca().invert_xaxis()
    # plt.clim([-0.5,0.5])
    # plt.show()


    contrast_5sig_1D_seps = np.arange(0,3,0.1)
    contrast_5sig_1D_med = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    contrast_5sig_1D_min = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    contrast_5sig_1D_inspike = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    contrast_5sig_1D_outspike = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    for k,sep in enumerate(contrast_5sig_1D_seps):
        # mask=copy(contrast_5sig_combined)
        # mask[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))] = np.nan
        # print( np.nanmedian(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))]))
        # plt.imshow(mask,origin="lower")
        # plt.show()
        contrast_5sig_1D_med[k] = np.nanmedian(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
        contrast_5sig_1D_min[k] = np.nanmin(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
        contrast_5sig_1D_inspike[k] = np.nanmedian(contrast_5sig_combined[np.where((~mask)*(rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
        contrast_5sig_1D_outspike[k] = np.nanmin(contrast_5sig_combined[np.where(mask*(rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
    fm_r2comp_grid = np.sqrt((nrs1_RA_grid-ra_offset)**2+(nrs1_Dec_grid-dec_offset)**2)


    rdi_nrs2_flux_filename = os.path.join(out_png,"HD19467B_RDI_F460M_nrs2_flux.fits")
    with fits.open(rdi_nrs2_flux_filename) as hdulist:
        rdi_nrs2_flux = hdulist[0].data*F460M_to_F444W
    rdi_nrs2_fluxerr_filename = os.path.join(out_png,"HD19467B_RDI_F460M_nrs2_fluxerr.fits")
    with fits.open(rdi_nrs2_fluxerr_filename) as hdulist:
        rdi_nrs2_fluxerr = hdulist[0].data*F460M_to_F444W
    rdi_nrs2_RADec_filename = os.path.join(out_png,"HD19467B_RDI_F460M_nrs2_RADec.fits")
    with fits.open(rdi_nrs2_RADec_filename) as hdulist:
        rdi_nrs2_RA_grid = hdulist[0].data - ra_corr
        rdi_nrs2_Dec_grid = hdulist[1].data - dec_corr
        rdi_nrs2_RA_vec =   rdi_nrs2_RA_grid[0,:]
        rdi_nrs2_Dec_vec =   rdi_nrs2_Dec_grid[:,0]
    rdi_dra = rdi_nrs2_RA_vec[1]-rdi_nrs2_RA_vec[0]
    rdi_ddec = rdi_nrs2_Dec_vec[1]-rdi_nrs2_Dec_vec[0]
    rdi_rs_grid = np.sqrt(rdi_nrs2_RA_grid**2+rdi_nrs2_Dec_grid**2)

    rdi_nrs1_flux_filename = os.path.join(out_png,"HD19467B_RDI_F360M_nrs1_flux.fits")
    with fits.open(rdi_nrs1_flux_filename) as hdulist:
        rdi_nrs1_flux = hdulist[0].data*F360M_to_F444W
    rdi_nrs1_fluxerr_filename = os.path.join(out_png,"HD19467B_RDI_F360M_nrs1_fluxerr.fits")
    with fits.open(rdi_nrs1_fluxerr_filename) as hdulist:
        rdi_nrs1_fluxerr = hdulist[0].data*F360M_to_F444W
    rdi_nrs1_RADec_filename = os.path.join(out_png,"HD19467B_RDI_F360M_nrs1_RADec.fits")
    with fits.open(rdi_nrs1_RADec_filename) as hdulist:
        rdi_nrs1_RA_grid = hdulist[0].data - ra_corr
        rdi_nrs1_Dec_grid = hdulist[1].data - dec_corr
        rdi_nrs1_RA_vec =   rdi_nrs1_RA_grid[0,:]
        rdi_nrs1_Dec_vec =   rdi_nrs1_Dec_grid[:,0]


    rdi_deno = (1/rdi_nrs1_fluxerr**2 + 1/rdi_nrs2_fluxerr**2)
    rdi_combined_flux = (rdi_nrs1_flux/rdi_nrs1_fluxerr**2 + rdi_nrs2_flux/rdi_nrs2_fluxerr**2)/rdi_deno
    kmax,lmax = np.unravel_index(np.nanargmax(rdi_combined_flux*(((rdi_nrs1_RA_grid-(-1.5))**2+(rdi_nrs1_Dec_grid-(-0.9))**2)<0.4)),rdi_combined_flux.shape)
    rdi_r2comp_grid = np.sqrt((rdi_nrs1_RA_grid-rdi_nrs1_RA_grid[kmax,lmax])**2+(rdi_nrs1_Dec_grid-rdi_nrs1_Dec_grid[kmax,lmax])**2)
    # rdi_combined_flux_err = 1/np.sqrt(rdi_deno)
    rdi_combined_SNR = rdi_combined_flux / ((rdi_nrs1_fluxerr+rdi_nrs2_fluxerr)/2.)
    rdi_combined_SNR = rdi_combined_SNR / np.nanstd(rdi_combined_SNR[np.where(rdi_r2comp_grid > 0.3)])
    rdi_combined_flux_err = rdi_combined_flux/rdi_combined_SNR
    # rdi_combined_SNR = rdi_combined_flux/rdi_combined_flux_err
    rdi_contrast_5sig_combined  = 5*rdi_combined_flux_err/HD19467_flux_MJy["F444W"]
    rdi_contrast_5sig_combined[np.where(~np.isfinite(rdi_contrast_5sig_combined))] = np.nan
    rdi_contrast_5sig_1D_seps = np.arange(0,3,0.1)
    rdi_contrast_5sig_1D_med = np.zeros(rdi_contrast_5sig_1D_seps.shape)+np.nan
    rdi_contrast_5sig_1D_min = np.zeros(rdi_contrast_5sig_1D_seps.shape)+np.nan
    for k,sep in enumerate(rdi_contrast_5sig_1D_seps):
        rdi_contrast_5sig_1D_med[k] = np.nanmedian(rdi_contrast_5sig_combined[np.where((rdi_rs_grid>(sep-0.05))*(rdi_rs_grid<(sep+0.05)))])
        rdi_contrast_5sig_1D_min[k] = np.nanmin(rdi_contrast_5sig_combined[np.where((rdi_rs_grid>(sep-0.05))*(rdi_rs_grid<(sep+0.05)))])



    # plt.show()
    fontsize=12
    if 1: #figure 1
        fig = plt.figure(1, figsize=(12,5))
        gs = gridspec.GridSpec(2,5, height_ratios=[0.05,1], width_ratios=[0.2,1,1,0.3,1])
        gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)

        fontsize=12
        # plt.figure(1,figsize=(12,3.25))
        # plt.subplot(1,3,1)
        ax1 = plt.subplot(gs[1, 1])
        plt1 = plt.imshow(HD19467_median_im[:,::-1]/1e-12,origin="lower",extent=[s3d_RA_vec[0]-s3d_dRA/2.,s3d_RA_vec[-1]+s3d_dRA/2.,s3d_Dec_vec[0]-s3d_dDec/2.,s3d_Dec_vec[-1]+s3d_dDec/2.])
        plt.text(0.03, 0.99, "HD 19467 AB\nMedian Cube", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        # plt.plot(0,0,"*",color=color_list[0],markersize=10)
        # plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color=color_list[0]) #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="pink",markersize=10)
        # plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="pink")# \n 36-166 $\mu$Jy
        plt.plot(0,0,"*",color="grey",markersize=10)
        txt = plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_aspect('equal')
        plt.gca().invert_xaxis()

        # plt.gca().annotate('HD19467 \n 1.1-2.6 Jy)', xy=(0, 0), xytext=(0+1.5, 0-1.5),color="white",
        #             arrowprops=dict(facecolor='white', shrink=0.05),
        #             )
        # plt.gca().annotate('HD19467B \n 36-166 $\mu$Jy)', xy=(ra_offset, dec_offset), xytext=(ra_offset+1.5, dec_offset-1.5),color="white",
        #             arrowprops=dict(facecolor='white', shrink=0.05),horizontalalignment='center',verticalalignment='top'
        #             )
        # cbar = plt.colorbar(pad=0)#mappable=CS,
        # cbar.ax.tick_params(labelsize=fontsize)
        # cbar.set_label(label='Flux ($\mu$Jy)', fontsize=fontsize)
        plt.clim([0,200])
        # cbar.set_label(label='log(Flux / (1 MJy))', fontsize=fontsize)
        # plt.clim([-14,-7])
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 1])
        cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0, 50,100,150,200])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        # plt.subplot(1,3,2)
        ax1 = plt.subplot(gs[1, 2])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt2 = plt.imshow(combined_SNR,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Companion S/N", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        txt = plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        if 1: # plot copass
            plt.text(1.5,-1.95, 'N', fontsize=fontsize, ha='center', va='bottom', color="black")  # \n 1.1-2.6 Jy
            plt.gca().annotate('', xy=(1.5,-2.), xytext=(1.5,-2.5),color="black", fontsize=fontsize,
                arrowprops=dict(facecolor="black",width=1),horizontalalignment='center',verticalalignment='center')
            plt.text(2.05,-2.5, 'E', fontsize=fontsize, ha='right', va='center', color="black")  # \n 1.1-2.6 Jy
            plt.gca().annotate('', xy=(2.0,-2.5), xytext=(1.5,-2.5),color="black", fontsize=fontsize,
                arrowprops=dict(facecolor="black",width=1),horizontalalignment='center',verticalalignment='center')#, shrink=0.001
        # cbar = plt.colorbar(pad=0)#mappable=CS,
        # cbar.ax.tick_params(labelsize=fontsize)
        # cbar.set_label(label='S/N', fontsize=fontsize)
        plt.clim([-2,10])
        # plt.xlim([-rad,rad])
        # plt.ylim([-rad,rad])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 2])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'S/N', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0,2,4,6,8,10])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        # plt.figure(2,figsize=(6,3.25))
        # plt.subplot(1,3,3)
        ax1 = plt.subplot(gs[1, 4])
        plt.scatter(rgrid_psfprofile.ravel(),im_psfprofile.ravel(),c="grey",s=20,label="PSF profile",marker="+",linewidths=1,alpha=1)
        # plt.scatter(webbpsf_R,wepsfs1[arg3um,:,:]/np.nanmax(wepsfs1[arg3um,:,:]),s=0.2,label="WebbPSF (3$\mu$m)",c="grey",alpha=0.3,marker="x")
        # plt.scatter(webbpsf_R,wepsfs2[arg5um,:,:]/np.nanmax(wepsfs2[arg5um,:,:]),s=0.2,label="WebbPSF (5$\mu$m)",c="grey",alpha=0.3,marker="+")
        # plt.scatter(webbpsf_R,wpsfs1[arg3um,:,:]/np.nanmax(wpsfs1[arg3um,:,:]),s=0.2,label="WebbPSF (3$\mu$m)",c="red",alpha=0.3,marker="x")
        # plt.scatter(webbpsf_R,wpsfs2[arg5um,:,:]/np.nanmax(wpsfs2[arg5um,:,:]),s=0.2,label="WebbPSF (5$\mu$m)",c="red",alpha=0.3,marker="+")
        plt.scatter(rs_grid,contrast_5sig_combined,s=0.2,c=color_list[0],alpha=0.2)
        plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_med,label="5$\sigma$ FM Median",alpha=1,c=color_list[0],linestyle="-")
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_min,label="5$\sigma$ FM Best",alpha=1,c=color_list[0],linestyle="--")
        # contrast_5sig_1D_inspike
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_inspike,label="5$\sigma$ FM inspike",alpha=1,c="red",linestyle="-")
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_outspike,label="5$\sigma$ FM outspike",alpha=1,c="red",linestyle="--")
        plt.scatter(rdi_rs_grid,rdi_contrast_5sig_combined,s=0.2,c=color_list[1],alpha=0.2)
        plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_med,label="5$\sigma$ RDI Median",alpha=1,c=color_list[1],linestyle="-.")
        # plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_min,label="5$\sigma$ RDI Best",alpha=1,c=color_list[1],linestyle=":")

        # plt.scatter(rs_grid,contrast_5sig_combined,s=0.2,label="5$\sigma$ - Forward Model",c=color_list[0],alpha=0.3)
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_med,label="5$\sigma$ - Forward Model - Median",alpha=1,c=color_list[0],linestyle="-")
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_min,label="5$\sigma$ - Forward Model - lower limit",alpha=1,c=color_list[0],linestyle="--")
        # plt.scatter(rdi_rs_grid,rdi_contrast_5sig_combined,s=0.2,label="5$\sigma$ - RDI",c=color_list[1],alpha=0.3)
        # plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_med,label="5$\sigma$ - RDI - Median",alpha=1,c=color_list[1],linestyle="-.")
        # plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_min,label="5$\sigma$ - RDI - lower limit",alpha=1,c=color_list[1],linestyle=":")

        # plt.plot(F444W_Carter_seps,F444W_Carter_cons*np.sqrt(1236/2100),alpha=1,c="black",linestyle="-")#,label="NIRCam ADI+RDI"
        plt.plot(F444W_Carter_seps,F444W_Carter_cons*np.sqrt(1236/2100*10**((5.4-6.771)/2.5)),alpha=1,c="black",linestyle="-")#,label="NIRCam ADI+RDI"
        plt.text(3.5,8e-7, 'NIRCam', fontsize=fontsize, ha='left', va='center',color="black",rotation=0)# \n 36-166 $\mu$Jy
        plt.fill_between([0,0.3],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.3)

        plt.plot(np.sqrt(ra_offset**2+dec_offset**2),HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"],".",color="black",markersize=10)
        plt.text(np.sqrt(ra_offset**2+dec_offset**2)+0.1,HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"], 'HD19467B', fontsize=fontsize, ha='left', va='center',color="black")# \n 36-166 $\mu$Jy

        plt.yscale("log")
        plt.xscale("log")
        # plt.xlim([0,3])
        plt.xlim([0.1,10])
        plt.ylim([10**(-6.3),10**(-1)])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Flux ratio ({0})".format("F444W"),fontsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=10)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)




        plt.tight_layout()
        out_filename = os.path.join(out_png, "summarydetec.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "summarydetec_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    if 1:
        fig = plt.figure(2, figsize=(12,4))
        gs = gridspec.GridSpec(2,6, height_ratios=[0.05,1], width_ratios=[0.2,1,1,1,0.35,1])
        gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)
        # gs.update(left=0.00, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)
        ax1 = plt.subplot(gs[1, 1])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt1 = plt.imshow(combined_flux/1e-12,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Flux (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.text(0.03, 0.01, "Forward Model (FM)", fontsize=1.4*fontsize, ha='left', va='bottom', transform=plt.gca().transAxes,color=color_list[0],zorder=10)
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        # plt.clim([-5,5])
        plt.clim([-50,50])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 1])
        cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        # cb.set_ticks([-5,-2.5,0,2.5])  # xticks[1::]
        cb.set_ticks([-50,-25,0,25])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[1, 2])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt2 = plt.imshow(np.log10(combined_flux_err/1e-12),origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Flux error (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-1.0,1.0])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 2])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([-1, 0,1])
        cb.set_ticklabels(["$10^{-1}$","$10^{0}$","$10^{1}$"])  # xticks[1::]
        # cb.set_ticklabels(["0.1","1","10"])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax3 = plt.subplot(gs[1, 3])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt2 = plt.imshow(combined_SNR,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Companion S/N", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-2,10])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 3])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'S/N', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0,2,4,6,8,10])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[0:2, 5])
        snr_map_masked = copy(combined_SNR)
        nan_mask_boxsize=5
        snr_map_masked[np.where(np.isnan(correlate2d(snr_map_masked,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        snr_map_masked[np.where((fm_r2comp_grid < 0.7))] = np.nan
        # plt.figure(10)
        # plt.imshow(snr_map_masked)
        # plt.show()
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3,range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="Combined",color=color_list[0],linestyle="-")
        plt.plot(nrs1_bins,nrs1_hist, label="NRS1",color=color_list[1],linestyle=":")
        plt.plot(nrs2_bins,nrs2_hist, label="NRS2",color="pink",linestyle="-.")

        plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
                 linestyle="--", label="Gaussian")
        plt.yscale("log")
        plt.xlim([-5, 5])
        plt.ylim([1e-4, 1])
        plt.xlabel("SNR",fontsize=fontsize)
        plt.ylabel("PDF",fontsize=fontsize)
        plt.legend()
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        out_filename = os.path.join(out_png, "FMdetec_summary.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "FMdetec_summary_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    if 1:
        fig = plt.figure(3, figsize=(12,4))
        gs = gridspec.GridSpec(2,6, height_ratios=[0.05,1], width_ratios=[0.2,1,1,1,0.35,1])
        gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)
        # gs.update(left=0.00, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)
        ax1 = plt.subplot(gs[1, 1])
        plt1 = plt.imshow(rdi_combined_flux/1e-12,origin="lower",extent=[rdi_nrs1_RA_vec[0]-rdi_dra/2.,rdi_nrs1_RA_vec[-1]+rdi_dra/2.,rdi_nrs1_Dec_vec[0]-rdi_ddec/2.,rdi_nrs1_Dec_vec[-1]+rdi_ddec/2.])
        plt.text(0.03, 0.99, "Flux (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.text(0.03, 0.01, "RDI", fontsize=1.5*fontsize, ha='left', va='bottom', transform=plt.gca().transAxes,color=color_list[1])
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-50,50])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 1])
        cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([-50,-25,0,25])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[1, 2])
        plt2 = plt.imshow(np.log10(rdi_combined_flux_err/1e-12),origin="lower",extent=[rdi_nrs1_RA_vec[0]-rdi_dra/2.,rdi_nrs1_RA_vec[-1]+rdi_dra/2.,rdi_nrs1_Dec_vec[0]-rdi_ddec/2.,rdi_nrs1_Dec_vec[-1]+rdi_ddec/2.])
        plt.text(0.03, 0.99, "Flux error (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([1.0,3.0])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 2])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([1, 2,3])
        cb.set_ticklabels(["$10^{1}$","$10^{2}$","$10^{3}$"])  # xticks[1::]
        # cb.set_ticklabels(["0.1","1","10"])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax3 = plt.subplot(gs[1, 3])
        plt2 = plt.imshow(rdi_combined_SNR,origin="lower",extent=[rdi_nrs1_RA_vec[0]-rdi_dra/2.,rdi_nrs1_RA_vec[-1]+rdi_dra/2.,rdi_nrs1_Dec_vec[0]-rdi_ddec/2.,rdi_nrs1_Dec_vec[-1]+rdi_ddec/2.])
        plt.text(0.03, 0.99, "Companion S/N", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-2,10])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 3])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'S/N', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0,2,4,6,8,10])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[0:2, 5])
        snr_map_masked = copy(rdi_combined_SNR)
        nan_mask_boxsize=5
        snr_map_masked[np.where(np.isnan(correlate2d(snr_map_masked,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        snr_map_masked[np.where((fm_r2comp_grid < 0.7))] = np.nan
        # plt.figure(10)
        # plt.imshow(snr_map_masked)
        # plt.show()
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3,range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="Combined",color=color_list[0],linestyle="-")
        # plt.plot(nrs1_bins,nrs1_hist, label="NRS1",color=color_list[1],linestyle=":")
        # plt.plot(nrs2_bins,nrs2_hist, label="NRS2",color="pink",linestyle="-.")

        plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
                 linestyle="--", label="Gaussian")
        plt.yscale("log")
        plt.xlim([-5, 5])
        plt.ylim([1e-4, 1])
        plt.xlabel("SNR",fontsize=fontsize)
        plt.ylabel("PDF",fontsize=fontsize)
        plt.legend()
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        out_filename = os.path.join(out_png, "RDIdetec_summary.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "RDIdetec_summary_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    if 1:
        # webbpsf_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/jw01414004001_02101_nrs1_webbpsf.fits"
        # print("loading" , webbpsf_filename)
        # with fits.open(webbpsf_filename) as hdulist:
        #     wpsfs1 = hdulist[0].data
        #     wpsfs_header = hdulist[0].header
        #     wepsfs1 = hdulist[1].data
        #     webbpsf1_wvs = hdulist[2].data
        #     webbpsf_X = hdulist[3].data
        #     webbpsf_Y = hdulist[4].data
        #     wpsf_pixelscale = wpsfs_header["PIXELSCL"]
        #     wpsf_oversample = wpsfs_header["oversamp"]
        # arg3um = np.argmin(np.abs(webbpsf1_wvs-3.0))
        # webbpsf_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/jw01414004001_02101_nrs2_webbpsf.fits"
        # print("loading" , webbpsf_filename)
        # with fits.open(webbpsf_filename) as hdulist:
        #     wpsfs2 = hdulist[0].data
        #     wpsfs_header = hdulist[0].header
        #     wepsfs2 = hdulist[1].data
        #     webbpsf2_wvs = hdulist[2].data
        #     webbpsf_X = hdulist[3].data
        #     webbpsf_Y = hdulist[4].data
        #     wpsf_pixelscale = wpsfs_header["PIXELSCL"]
        #     wpsf_oversample = wpsfs_header["oversamp"]
        # arg5um = np.argmin(np.abs(webbpsf2_wvs-5.0))
        # # wpsf_profile  = (np.nansum(wepsfs1,axis=0)+np.nansum(wepsfs2,axis=0))/2.
        # # print(wpsf_profile.shape)
        # webbpsf_R = np.sqrt(webbpsf_X ** 2 + webbpsf_Y ** 2)
        # print(webbpsf_R.shape)

        fig = plt.figure(4, figsize=(6,5))
        plt.scatter(rgrid_psfprofile.ravel(),im_psfprofile.ravel(),c="grey",s=20,label="PSF profile",marker="+",linewidths=1,alpha=1)
        # plt.scatter(webbpsf_R,wepsfs1[arg3um,:,:]/np.nanmax(wepsfs1[arg3um,:,:]),s=0.2,label="WebbPSF (3$\mu$m)",c="grey",alpha=0.3,marker="x")
        # plt.scatter(webbpsf_R,wepsfs2[arg5um,:,:]/np.nanmax(wepsfs2[arg5um,:,:]),s=0.2,label="WebbPSF (5$\mu$m)",c="grey",alpha=0.3,marker="+")
        # plt.scatter(webbpsf_R,wpsfs1[arg3um,:,:]/np.nanmax(wpsfs1[arg3um,:,:]),s=0.2,label="WebbPSF (3$\mu$m)",c="red",alpha=0.3,marker="x")
        # plt.scatter(webbpsf_R,wpsfs2[arg5um,:,:]/np.nanmax(wpsfs2[arg5um,:,:]),s=0.2,label="WebbPSF (5$\mu$m)",c="red",alpha=0.3,marker="+")
        plt.scatter(rs_grid,contrast_5sig_combined,s=0.2,c=color_list[0],alpha=0.3)#,label="5$\sigma$ - Forward Model"
        plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_med,label="5$\sigma$ - Forward Model - Median",alpha=1,c=color_list[0],linestyle="-")
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_min,label="5$\sigma$ - Forward Model - Best",alpha=1,c=color_list[0],linestyle="--")
        plt.scatter(rdi_rs_grid,rdi_contrast_5sig_combined,s=0.2,c=color_list[1],alpha=0.3)#,label="5$\sigma$ - RDI"
        plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_med,label="5$\sigma$ - RDI - Median",alpha=1,c=color_list[1],linestyle="-.")
        # plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_min,label="5$\sigma$ - RDI - Best",alpha=1,c=color_list[1],linestyle=":")
        plt.plot(F444W_Carter_seps,F444W_Carter_cons*np.sqrt(1236/2100),alpha=1,c="black",linestyle="-.")#,label="NIRCam ADI+RDI"
        plt.text(1.2,4e-6, 'NIRCam Carter+2023', fontsize=fontsize, ha='left', va='center',color="black",rotation=-11)# \n 36-166 $\mu$Jy
        plt.fill_between([0,0.3],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.2)

        plt.plot(np.sqrt(ra_offset**2+dec_offset**2),HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"],".",color="black",markersize=10)
        plt.text(np.sqrt(ra_offset**2+dec_offset**2)+0.1,HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"], 'HD19467B', fontsize=fontsize, ha='left', va='center',color="black")# \n 36-166 $\mu$Jy

        plt.yscale("log")
        plt.xscale("log")
        # plt.xlim([0,3])
        plt.xlim([0.1,10])
        plt.ylim([10**(-6.3),10**(-0)])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Flux ratio ({0})".format("F444W"),fontsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=10)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        out_filename = os.path.join(out_png, "contrasts.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "contrasts_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    plt.show()
