import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
import h5py
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patheffects as PathEffects


def fill_gap(HD19467B_wvs,HD19467B_spec,planet_f,fit_range= [3.9,4.3]):
    """
    Fit a model (planet_f) to fill the gap in a NIRSpec spectrum caused by the gap between NRS1 and NRS2.
    Returns new spectrum with added datapoints in the gap from the model fit.
    This is for photometric calculation.

    Args:
        HD19467B_wvs:
        HD19467B_spec:
        planet_f: interp1d interpolator object
        fit_range:

    Returns:

    """
    where_finite = np.where(np.isfinite(HD19467B_spec))
    if np.size(where_finite[0]) == 0:
        return None, None
    HD19467B_wvs = HD19467B_wvs[where_finite]
    HD19467B_spec = HD19467B_spec[where_finite]

    HD19467B_dwvs = HD19467B_wvs[1::]-HD19467B_wvs[0:np.size(HD19467B_wvs)-1]
    HD19467B_dwvs = np.insert(HD19467B_dwvs,0,HD19467B_dwvs[0])



    if np.size(np.where(HD19467B_dwvs>0.05)[0]) == 0:
        return None, None
    else:
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

    ####################
    ## To be modified
    ####################
    # Ouput dir for final figures and fits files
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/external/"
    # List of colors used in the plots
    color_list = ["#ff9900","#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
    fitpsf_filename_nrs1 = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils/jw01414004001_02101_nrs1_fitpsf_webbpsf.fits"
    fitpsf_filename_nrs2 = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils/jw01414004001_02101_nrs2_fitpsf_webbpsf.fits"
    # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
    wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
    wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
    # Absolute fluxes for the host star to be used in calculated flux ratios with the companion.
    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}
    ####################


    if 1: # Photometry of the host star HD19467A
        HD19467A_wvs = np.concatenate([wv_sampling_nrs1,wv_sampling_nrs2])
        with fits.open(fitpsf_filename_nrs1) as hdulist:
            bestfit_coords1 = hdulist[0].data
        with fits.open(fitpsf_filename_nrs2) as hdulist:
            bestfit_coords2 = hdulist[0].data
        HD19467A_spec = np.concatenate([bestfit_coords1[0,:,0],bestfit_coords2[0,:,0]])
        where_good = np.where(np.isfinite(HD19467A_spec)*((HD19467A_wvs<4.02)+(HD19467A_wvs>4.22)))
        HD19467A_wvs = HD19467A_wvs[where_good]
        HD19467A_spec = HD19467A_spec[where_good]
        z = np.polyfit(HD19467A_wvs[np.where(np.isfinite(HD19467A_spec))], HD19467A_spec[np.where(np.isfinite(HD19467A_spec))], 2)
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
        for photfilter_name in ["F360M", "F410M", "F430M", "F460M", "F444W", "F356W","Lp","Mp"]:  # HD19467_flux_MJy.keys():
            if photfilter_name == "Lp":
                photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/Keck_NIRC2.Lp.dat"
            elif photfilter_name == "Mp":
                photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/Paranal_NACO.Mp.dat"
            else:
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
        HD19467_flux_MJy["Lp"] = HD19467_flux_MJy["F360M"] / HD19467A_NIRSpec_flux_MJy["F360M"] * HD19467A_NIRSpec_flux_MJy["Lp"]
        HD19467_flux_MJy["Mp"] = HD19467_flux_MJy["F360M"] / HD19467A_NIRSpec_flux_MJy["F360M"] * HD19467A_NIRSpec_flux_MJy["Mp"]

        print(HD19467_flux_MJy)

    # NIRCam photometry from Jens Kammerer
    if 1:
        HD19467B_Jens_flux_MJy = {}
        HD19467B_Jens_fluxerr_MJy = {}
        HD19467B_Jens_flux_Wmum = {}
        HD19467B_Jens_fluxerr_Wmum = {}
        for photfilter_name in ["F250M","F300M","F360M","F410M","F430M","F460M"]:
            Jens_filename = os.path.join(external_dir,"20240205_Jens_Nircam_HD19467B/JWST_NIRCAM_NRCALONG_" + photfilter_name + "_MASKBAR_MASKLWB_SUB320ALWB-fitpsf_c1.fits")
            with fits.open(Jens_filename) as hdulist:
                myheader = hdulist[0].header
            HD19467B_Jens_flux_MJy[photfilter_name] = float(myheader["FLUX_JY"])*1e-6
            HD19467B_Jens_fluxerr_MJy[photfilter_name] = float(myheader["FLUX_JY_ERR"])*1e-6
            HD19467B_Jens_flux_Wmum[photfilter_name] = float(myheader["FLUX_SI"])
            HD19467B_Jens_fluxerr_Wmum[photfilter_name] = float(myheader["FLUX_SI_ERR"])

            #F250M & $18.4\pm0.9$ & $0.88\pm0.04$ && $13.23\pm0.05$ & 0.918 & 0.963
            print(photfilter_name+" & "+\
                  "${0:.1f}\\pm{1:.1f}$".format(float(myheader["FLUX_JY"])*1e6,float(myheader["FLUX_JY_ERR"])*1e6)+" & "+\
                  "${0:.2f}\\pm{1:.2f}$".format(float(myheader["FLUX_SI"])*1e17,float(myheader["FLUX_SI_ERR"])*1e17)+" & "+\
                  " {0:.2f} (10\\%)".format(0.1*float(myheader["FLUX_SI"])*1e17)+" & "+\
                  "${0:.2f}\\pm{1:.2f}$".format(float(myheader["DELMAG"]),float(myheader["DELMAG_ERR"]))+" & "+\
                  "{0:.3f}".format(float(myheader["TP_CORONMSK"]))+" & "+\
                  "{0:.3f}".format(float(myheader["TP_COMSUBST"]))+\
                  "\\\\")

    # Read a BTsettl grid to get a model that will be used to fill the gap in the spectrum
    if 1:
        photfilter = os.path.join(external_dir,"JWST_NIRCam." + "F460M" + ".dat")
        filter_arr = np.loadtxt(photfilter)
        trans_wvs = filter_arr[:, 0] / 1e4
        trans = filter_arr[:, 1]
        photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
        photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
        with h5py.File(os.path.join(external_dir, "BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"), 'r') as hf:
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
        myinterpgrid = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear", bounds_error=False,
                                               fill_value=np.nan)
        best_fit_planet_model = myinterpgrid((800, 5.0))
        best_fit_planet_model = (best_fit_planet_model * (u.W * u.m ** -2 / u.um) * (grid_wvs * u.um) ** 2 / const.c).to(u.MJy).value
        planet_f = interp1d(grid_wvs, best_fit_planet_model, bounds_error=False, fill_value=0)

    if 1: #photometry from NIRSpec RDI spectrum
        RDI_1dspectrum_speckles_filename = os.path.join(out_png, "HD19467b_RDI_1dspectrum_speckles_MJy.fits")
        with fits.open(RDI_1dspectrum_speckles_filename) as hdulist:
            HD19467B_speckles_wvs = hdulist[0].data
            HD19467B_speckles_spec = hdulist[1].data
            speckles_median_smooth = hdulist[2].data
        HD19467B_speckles_spec = HD19467B_speckles_spec - speckles_median_smooth[:,None]
        where_nonempty = np.where(np.nansum(np.isfinite(HD19467B_speckles_spec),axis=0)!=0)
        HD19467B_speckles_spec = HD19467B_speckles_spec[:,where_nonempty[0]]

        RDI_1dspectrum_filename = os.path.join(out_png, "HD19467b_RDI_1dspectrum_MJy.fits")
        with fits.open(RDI_1dspectrum_filename) as hdulist:
            HD19467B_wvs = hdulist[0].data
            HD19467B_spec = hdulist[1].data
            HD19467B_spec_err = hdulist[2].data
            HD19467B_spec_errhpf = hdulist[3].data
        # HD19467B_spec[np.where(np.isnan(HD19467B_spec)+(HD19467B_spec_errhpf>1.5e-11))] = np.nan
        # HD19467B_spec_errhpf[np.where(np.isnan(HD19467B_spec)+(HD19467B_spec_errhpf>1.5e-11))] = np.nan
        # HD19467B_spec_err[np.where(np.isnan(HD19467B_spec)+(HD19467B_spec_errhpf>1.5e-11))] = np.nan

        HD19467B_dwvs = HD19467B_wvs[1::]-HD19467B_wvs[0:np.size(HD19467B_wvs)-1]
        HD19467B_dwvs = np.insert(HD19467B_dwvs,0,HD19467B_dwvs[0])

        if 1: #fill gap
            HD19467B_gapfilled_wvs,HD19467B_gapfilled_spec = fill_gap(HD19467B_wvs,HD19467B_spec,planet_f)
            HD19467B_gapfilled_dwvs = HD19467B_gapfilled_wvs[1::]-HD19467B_gapfilled_wvs[0:np.size(HD19467B_gapfilled_wvs)-1]
            HD19467B_gapfilled_dwvs = np.insert(HD19467B_gapfilled_dwvs,0,HD19467B_gapfilled_dwvs[0])

        HD19467B_NIRSpec_flux_MJy = {}
        HD19467B_NIRSpec_fluxerr_MJy = {}
        HD19467B_NIRSpec_flux_Wm2um={}
        HD19467B_NIRSpec_fluxerr_Wm2um={}
        for photfilter_name in ["F360M","F410M","F430M","F460M","F444W","F356W","Lp","Mp"]:#HD19467_flux_MJy.keys():
            if photfilter_name == "Lp":
                photfilter = os.path.join(external_dir, "Keck_NIRC2.Lp.dat")
            elif photfilter_name == "Mp":
                photfilter = os.path.join(external_dir, "Paranal_NACO.Mp.dat")
            else:
                photfilter = os.path.join(external_dir, "JWST_NIRCam." + photfilter_name + ".dat")
            filter_arr = np.loadtxt(photfilter)
            trans_wvs = filter_arr[:, 0] / 1e4
            trans = filter_arr[:, 1]
            photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
            photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
            bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
            photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]

            filter_norm = np.nansum((HD19467B_gapfilled_dwvs * u.um) * photfilter_f(HD19467B_gapfilled_wvs))

            aper_spec_Flambda = (HD19467B_gapfilled_spec * u.MJy * const.c / (HD19467B_gapfilled_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um)
            Flambda = np.nansum((HD19467B_gapfilled_dwvs * u.um) * photfilter_f(HD19467B_gapfilled_wvs) * (aper_spec_Flambda)) / filter_norm
            Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
            Fnu_Mjy = Fnu.to(u.MJy).value
            Flambda_Wm2um = Flambda.to(u.W * u.m ** -2 / u.um).value

            Fnu_MJy_speckles = []
            Flambda_Wm2um_speckles = []
            for speckle_spec in HD19467B_speckles_spec.T:
                speckle_gapfilled_wvs,speckle_gapfilled_spec = fill_gap(HD19467B_wvs,speckle_spec,planet_f)
                if speckle_gapfilled_spec is None:
                    continue
                speckle_gapfilled_dwvs = speckle_gapfilled_wvs[1::]-speckle_gapfilled_wvs[0:np.size(speckle_gapfilled_wvs)-1]
                speckle_gapfilled_dwvs = np.insert(speckle_gapfilled_dwvs,0,speckle_gapfilled_dwvs[0])

                filter_norm = np.nansum((speckle_gapfilled_dwvs * u.um) * photfilter_f(speckle_gapfilled_wvs))
                aper_spec_Flambda = (speckle_gapfilled_spec * u.MJy * const.c / (speckle_gapfilled_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um)
                Flambda = np.nansum((speckle_gapfilled_dwvs * u.um) * photfilter_f(speckle_gapfilled_wvs) * (aper_spec_Flambda)) / filter_norm
                Flambda_Wm2um_speckles.append(Flambda.to(u.W * u.m ** -2 / u.um).value)
                Fnu_speckle = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
                Fnu_MJy_speckles.append(Fnu_speckle.to(u.MJy).value)
            Fnu_err_Mjy = np.nanstd(Fnu_MJy_speckles)
            Flambda_err_Wm2um = np.nanstd(Flambda_Wm2um_speckles)

            # print(photfilter_name, "{0}+-{1} MJy ({2:.1f}-{3:.1f}um)".format(Fnu_Mjy,Fnu_err_Mjy,photfilter_wvmin, photfilter_wvmax))
            # print(photfilter_name, "{0}+-{1} W/m2/um ({2:.1f}-{3:.1f}um)".format(Flambda_Wm2um,Flambda_err_Wm2um,photfilter_wvmin, photfilter_wvmax))
            mag = -2.5*np.log10(Fnu_Mjy/HD19467_flux_MJy[photfilter_name])
            mag_max = -2.5*np.log10((Fnu_Mjy+Fnu_err_Mjy)/HD19467_flux_MJy[photfilter_name])
            mag_min = -2.5*np.log10((Fnu_Mjy-Fnu_err_Mjy)/HD19467_flux_MJy[photfilter_name])
            # print(photfilter_name, "{0}+{1}-{2} W/m2/um".format(mag,mag_max-mag,mag-mag_min))
            HD19467B_NIRSpec_flux_MJy[photfilter_name] = Fnu_Mjy
            HD19467B_NIRSpec_fluxerr_MJy[photfilter_name] = Fnu_err_Mjy
            HD19467B_NIRSpec_flux_Wm2um[photfilter_name] = Flambda_Wm2um
            HD19467B_NIRSpec_fluxerr_Wm2um[photfilter_name] = Flambda_err_Wm2um


            #F250M & $18.4\pm0.9$ & $0.88\pm0.04$ && $13.23\pm0.05$ & 0.918 & 0.963
            print(photfilter_name+" & "+\
                  "${0:.1f}\\pm{1:.1f}$".format(Fnu_Mjy*1e12,Fnu_err_Mjy*1e12)+" & "+\
                  "${0:.2f}\\pm{1:.2f}$".format(Flambda_Wm2um*1e17,Flambda_err_Wm2um*1e17)+" & "+\
                  " {0:.2f} (10\\%)".format(0.1*Flambda_Wm2um*1e17)+" & "+\
                  "${0:.2f}^{1:.2f}_{1:.2f}$".format(mag,mag_max-mag,mag-mag_min)+" & "+\
                  "-"+" & "+\
                  "-"+\
                  "\\\\")

        print(HD19467B_NIRSpec_flux_MJy)
        print(HD19467B_NIRSpec_fluxerr_MJy)

    # Plot figure 3
    if 1:
        fontsize=12
        plt.figure(2,figsize=(12,8))
        for nrs_id,(lmin,lmax) in enumerate([(2.85,4.25),(4.05,5.3)]):
            plt.subplot(2,1,nrs_id+1)

            height = 3.5e-17
            width = 0.4e-17
            if nrs_id==0:
                wmin,wmax = 3.2,3.6
                plt.fill_between([wmin,wmax],[height,height],[height+width, height+width],color="grey",alpha=0.5, lw=0)
                txt = plt.text((wmin+wmax)/2.0,height+width/2.0, 'CH$_4$', fontsize=fontsize, ha='center', va='center',color="black")# \n 36-166 $\mu$Jy
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                wmin,wmax = 2.8,3.0
                plt.fill_between([wmin,wmax],[height,height],[height+width, height+width],color="grey",alpha=0.5, lw=0)
                txt = plt.text((wmin+wmax)/2.0,height+width/2.0, 'H$_2$O', fontsize=fontsize, ha='center', va='center',color="black")# \n 36-166 $\mu$Jy
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
            if nrs_id==1:
                wmin,wmax = 4.22,4.4
                plt.fill_between([wmin,wmax],[height,height],[height+width, height+width],color="grey",alpha=0.5, lw=0)
                txt = plt.text((wmin+wmax)/2.0,height+width/2.0, 'CO$_2$', fontsize=fontsize, ha='center', va='center',color="black")# \n 36-166 $\mu$Jy
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                wmin,wmax = 4.45,4.9
                plt.fill_between([wmin,wmax],[height,height],[height+width, height+width],color="grey",alpha=0.5, lw=0)
                txt = plt.text((wmin+wmax)/2.0,height+width/2.0, 'CO', fontsize=fontsize, ha='center', va='center',color="black")# \n 36-166 $\mu$Jy
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])


            HD19467B_spec_Flambda = (HD19467B_spec * u.MJy * const.c / (HD19467B_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um).value
            HD19467B_spec_err_Flambda = (HD19467B_spec_err * u.MJy * const.c / (HD19467B_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um).value
            HD19467B_spec_errhpf_Flambda = (HD19467B_spec_errhpf * u.MJy * const.c / (HD19467B_wvs * u.um) ** 2).to(u.W * u.m ** -2 / u.um).value
            # plt.figure(10)
            # plt.plot(HD19467B_wvs,HD19467B_spec_Flambda/HD19467B_spec_err_Flambda)
            # plt.ylabel("S/N per bin")
            # plt.show()
            plt.text(0.01, 0.99, 'HD 19467 B', fontsize=fontsize, ha='left', va='top', color="black",transform=plt.gca().transAxes)
            plt.fill_between(HD19467B_wvs,-HD19467B_spec_err_Flambda,+HD19467B_spec_err_Flambda,color=color_list[1],alpha=0.5, lw=0,label="RDI error")
            plt.fill_between(HD19467B_wvs,-HD19467B_spec_errhpf_Flambda,+HD19467B_spec_errhpf_Flambda,color=color_list[0],alpha=1, lw=0,label="HPF error")
            plt.plot(HD19467B_wvs,HD19467B_spec_Flambda,color=color_list[1],label="RDI spectrum",linewidth=1)

            if nrs_id == 0:
                photlabel_list = ["F360M"]
            if nrs_id == 1:
                photlabel_list = ["F410M","F430M","F460M"]
            for photfilter_name in ["F360M","F410M","F430M","F460M","Lp"]:
                if photfilter_name != "Lp":
                    photfilter = os.path.join(external_dir,"JWST_NIRCam." + photfilter_name + ".dat")
                else:
                    photfilter = os.path.join(external_dir,"Keck_NIRC2.Lp.dat")
                filter_arr = np.loadtxt(photfilter)
                trans_wvs = filter_arr[:, 0] / 1e4
                trans = filter_arr[:, 1]
                photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
                photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
                bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
                photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]

                eb = plt.errorbar([photfilter_wv0], HD19467B_NIRSpec_flux_Wm2um[photfilter_name],
                             xerr=np.array([[photfilter_wv0 - photfilter_wvmin], [photfilter_wvmax - photfilter_wv0]]),
                             yerr=HD19467B_NIRSpec_fluxerr_Wm2um[photfilter_name],
                             color="#00ccff")  # ,label=photfilter_name
                eb[-1][0].set_linestyle('-')
                plt.scatter(photfilter_wv0,HD19467B_NIRSpec_flux_Wm2um[photfilter_name],color="#00ccff",s=100,marker="x",zorder=10)
                if photfilter_name in photlabel_list:
                    if photfilter_name == "F360M" or photfilter_name == "F430M":
                        txt = plt.text(photfilter_wv0-0.01,1.05*HD19467B_NIRSpec_flux_Wm2um[photfilter_name], photfilter_name, fontsize=fontsize, ha='right', va='bottom', color="black")
                        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                    elif photfilter_name == "F460M":
                        txt = plt.text(photfilter_wv0-0.01,1.1*HD19467B_NIRSpec_flux_Wm2um[photfilter_name], photfilter_name, fontsize=fontsize, ha='right', va='bottom', color="black")
                        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                    else:
                        txt = plt.text(photfilter_wv0+0.01,1.05*HD19467B_NIRSpec_flux_Wm2um[photfilter_name], photfilter_name, fontsize=fontsize, ha='left', va='bottom', color="black")
                        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
            # eb = plt.errorbar([photfilter_wv0], HD19467B_NIRSpec_flux_MJy[photfilter_name],
            #              xerr=np.array([[photfilter_wv0 - photfilter_wvmin], [photfilter_wvmax - photfilter_wv0]]),
            #              yerr=HD19467B_NIRSpec_fluxerr_MJy[photfilter_name],
            #              color="#00ccff" )
            # eb[-1][0].set_linestyle('-')
            plt.scatter(photfilter_wv0,HD19467B_NIRSpec_flux_Wm2um[photfilter_name],color="#00ccff",s=100,marker="x",label="NIRSpec RDI",zorder=10)

            for photfilter_name in ["F360M","F410M","F430M","F460M"]:
                photfilter = os.path.join(external_dir,"JWST_NIRCam." + photfilter_name + ".dat")
                filter_arr = np.loadtxt(photfilter)
                trans_wvs = filter_arr[:, 0] / 1e4
                trans = filter_arr[:, 1]
                photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
                photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
                bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
                photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]

                eb = plt.errorbar([photfilter_wv0], HD19467B_Jens_flux_Wmum[photfilter_name],
                             xerr=np.array([[photfilter_wv0 - photfilter_wvmin], [photfilter_wvmax - photfilter_wv0]]),
                             yerr=HD19467B_Jens_fluxerr_Wmum[photfilter_name],
                             color="#ffcc00")  # ,label=photfilter_name
                plt.scatter(photfilter_wv0,HD19467B_Jens_flux_Wmum[photfilter_name],color="#ffcc00",s=100,marker="^",zorder=10)
                eb[-1][0].set_linestyle('--')
            # eb = plt.errorbar([photfilter_wv0], HD19467B_Jens_flux_MJy[photfilter_name],
            #              xerr=np.array([[photfilter_wv0 - photfilter_wvmin], [photfilter_wvmax - photfilter_wv0]]),
            #              yerr=HD19467B_Jens_fluxerr_MJy[photfilter_name],
            #              color="#ffcc00")
            # eb[-1][0].set_linestyle('--')
            plt.scatter(photfilter_wv0,HD19467B_Jens_flux_Wmum[photfilter_name],color="#ffcc00",s=100,marker="^" ,label="NIRCam ADI",zorder=10)

            photfilter = os.path.join(external_dir,"Keck_NIRC2.Lp.dat")
            filter_arr = np.loadtxt(photfilter)
            trans_wvs = filter_arr[:, 0] / 1e4
            trans = filter_arr[:, 1]
            photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
            photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
            bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
            photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]
            # 0.288±0.037 ×10-16 W m-2 µm-1
            Flambda = 0.288e-16 #* u.W * u.m ** -2 / u.um
            Flambda_min = (0.288e-16-0.037e-16)# * u.W * u.m ** -2 / u.um
            Flambda_max = (0.288e-16+0.037e-16) #* u.W * u.m ** -2 / u.um
            eb = plt.errorbar([photfilter_wv0], Flambda,
                         xerr=np.array([[photfilter_wv0 - photfilter_wvmin], [photfilter_wvmax - photfilter_wv0]]),
                         yerr=np.array([[Flambda - Flambda_min], [Flambda_max-Flambda]]),
                         color="black" )
            plt.scatter(photfilter_wv0,Flambda,color="black",label="Maire+2020",s=100,marker="*",zorder=10)
            eb[-1][0].set_linestyle('-.')
            if nrs_id == 0:
                txt = plt.text(photfilter_wv0+0.01,1.05*Flambda, "Lp", fontsize=fontsize, ha='left', va='bottom', color="black")
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


            plt.xlim([lmin,lmax])
            plt.ylim([-0.7e-17,5.5e-17])
            if nrs_id == 1:
                plt.legend(loc="upper right",handlelength=3)
            plt.ylabel("Flux (W/m$^2$/$\mu$m)",fontsize=fontsize)
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)

        plt.tight_layout()
        out_filename = os.path.join(out_png, "summaryRDIspec_Flambda.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "summaryRDIspec_Flambda_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        plt.show()