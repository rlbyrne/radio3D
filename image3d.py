import numpy as np
from datetime import datetime, timedelta
import spiceypy
import healpy
import scipy


def get_spacecraft_positions():
    # Function by Marin Anderson

    # SunRISE SPICE kernels and code originally provided by Jim Lux via LFT
    spiceypy.furnsh("MetaK.txt")
    """
    -2020 = constellation center / reference orbit
    -218 = spike / SRSP
    -190 = ein / SREI
    -191 = edward / SRED
    -225 = bebop / SRBB
    -186 = jet / SRJT
    -180 = faye / SRFY
    """
    objids = ["-2020", "-180", "-190", "-218", "-186", "-191", "-225"]

    def getpos(utctime):
        # convert UTC time to number of TDB seconds past the J2000 epoch
        et = spiceypy.str2et(utctime)
        calet = spiceypy.etcal(et)
        outpos = np.zeros((len(objids), 3))
        for ind, obj in enumerate(objids):
            # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/spkpos.html
            outpos[ind, :] = spiceypy.spkpos(obj, et, "J2000", "NONE", "-2020")[
                0
            ]  # spacecraft coordinates relative to constellation center
            # outpos[ind,:] = spiceypy.spkpos(obj,et,'J2000','NONE','EARTH')[0]  # spacecraft coordinates relative to Earth center
        return outpos

    timearr = np.arange(
        datetime(2024, 9, 27), datetime(2024, 9, 28), timedelta(minutes=1)
    ).astype(np.datetime64)
    outpos_all = np.zeros((len(objids), 3, len(timearr)))
    for ind, timeval in enumerate(timearr):
        outpos_all[:, :, ind] = getpos(str(timeval))

    return outpos_all


def get_baselines():

    outpos_all = get_spacecraft_positions()
    n_spacecraft = np.shape(outpos_all)[0]
    n_times = np.shape(outpos_all)[2]

    n_bls = int((n_spacecraft**2 - n_spacecraft) / 2)
    n_blts = n_bls * n_times
    ant1_inds = np.zeros(n_bls, dtype=int)
    ant2_inds = np.zeros(n_bls, dtype=int)
    bl_ind = 0
    for ant1_ind in range(n_spacecraft):
        for ant2_ind in range(ant1_ind + 1, n_spacecraft):
            ant1_inds[bl_ind] = ant1_ind
            ant2_inds[bl_ind] = ant2_ind
            bl_ind += 1
    time_inds = np.repeat(np.arange(n_times), n_bls)
    ant1_inds = np.tile(ant1_inds, n_times)
    ant2_inds = np.tile(ant2_inds, n_times)

    bl_coords = np.zeros((3, n_blts))
    for blt_ind in range(n_blts):
        bl_coords[:, blt_ind] = (
            outpos_all[ant1_inds[blt_ind], :, time_inds[blt_ind]]
            - outpos_all[ant2_inds[blt_ind], :, time_inds[blt_ind]]
        )

    return bl_coords


def get_pixel_coords(nside):

    npix = healpy.nside2npix(nside)
    theta, phi = healpy.pixelfunc.pix2ang(
        nside, np.arange(npix), nest=False
    )  # theta is the polar angle, phi is the azimuthal angle
    # Convert to Cartesian coordinates
    pixel_coords = np.zeros((3, npix), dtype=float)
    pixel_coords[0, :] = np.sin(theta) * np.cos(phi)
    pixel_coords[1, :] = np.sin(theta) * np.sin(phi)
    pixel_coords[2, :] = np.cos(theta)
    return pixel_coords  # Shape (3, npix,)


def simulate_visibilities(
    bl_coords,  # Shape (3, n_blts,)
    pixel_vals,  # Shape (npix,)
):
    n_blts = np.shape(bl_coords)[1]
    nside = healpy.npix2nside(len(pixel_vals))
    pixel_coords = get_pixel_coords(nside)
    visibilities = np.zeros(n_blts, dtype=complex)
    for pixel_ind in np.where(pixel_vals != 0):
        visibilities += pixel_vals[pixel_ind] * np.exp(
            2 * np.pi * 1j * np.sum(pixel_coords[:, pixel_ind] * bl_coords, axis=0)
        )
    return visibilities


def spherical_harmonic_imaging(
    visibilities,
    bl_coords,
    l_max,
):
    l_vals, m_vals = healpy.sphtfunc.Alm.getlm(l_max)
    alms = np.zeros(len(l_vals), dtype=complex)
    for blt_ind in range(len(visibilities)):
        bl_length = np.sqrt(np.sum(np.abs(bl_coords[:, blt_ind]) ** 2.0))
        theta = np.arccos(bl_coords[2, blt_ind] / bl_length)  # Polar angle
        phi = np.sign(bl_coords[1, blt_ind]) * np.arccos(
            bl_coords[0, blt_ind]
            / np.sqrt(
                np.abs(bl_coords[0, blt_ind]) ** 2.0
                + np.abs(bl_coords[1, blt_ind]) ** 2.0
            )
        )  # Azimuthal angle
        alms += (
            np.conj(
                (-1j) ** l_vals
                * scipy.special.spherical_jn(l_vals, bl_length)
                * scipy.special.sph_harm(m_vals, l_vals, phi, theta)
            )
            * visibilities[blt_ind]
        )
        +np.conj(
            (-1j) ** l_vals
            * scipy.special.spherical_jn(l_vals, bl_length)
            * scipy.special.sph_harm(m_vals, l_vals, phi + np.pi, np.pi - theta)
        ) * np.conj(visibilities[blt_ind])

    return alms, l_vals, m_vals


def pixel_based_imaging(
    visibilities,
    bl_coords,
    nside,
):
    pixel_coords = get_pixel_coords(nside)
    pixel_vals = np.zeros(healpy.nside2npix(nside), dtype=complex)
    for visiblity_ind in range(len(visibilities)):
        pixel_vals += 2 * np.real(
            visibilities[visiblity_ind]
            * np.exp(
                -2
                * np.pi
                * 1j
                * np.sum(pixel_coords * bl_coords[:, visiblity_ind, np.newaxis], axis=0)
            )
        )
    return pixel_vals


def alms_to_map(
    alms,
    l_vals,
    m_vals,
    nside,
    mirror_m=False,
):
    npix = healpy.nside2npix(nside)
    theta, phi = healpy.pixelfunc.pix2ang(nside, np.arange(npix), nest=False)
    pixel_values = np.zeros(npix, dtype=complex)
    for spherical_harm_ind in range(len(alms)):
        pixel_values += alms[spherical_harm_ind] * scipy.special.sph_harm(
            m_vals[spherical_harm_ind], l_vals[spherical_harm_ind], phi, theta
        )
        if mirror_m:
            if m_vals[spherical_harm_ind] != 0:
                pixel_values += np.conj(
                    alms[spherical_harm_ind]
                ) * scipy.special.sph_harm(
                    -m_vals[spherical_harm_ind], l_vals[spherical_harm_ind], phi, theta
                )
    return pixel_values


def alms_to_map_healpy(
    alms,  # Needs to be ordered the same as healpy.sphtfunc.Alm.getlm(l_max)
    nside,
):
    pixel_vals = healpy.sphtfunc.alm2map(alms, nside)
    return pixel_vals
