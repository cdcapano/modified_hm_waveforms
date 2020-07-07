# Copyright (C) 2020  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import (absolute_import, division)

import shlex
import numpy
from pycbc import conversions
from pycbc.waveform.waveform import (get_fd_waveform, props, NoWaveformError)


def modhm_fd(**kwargs):
    r"""Allows a waveform to be generated with different parameters for the
    sub-dominant modes.

    Modified parameters may be specified in one of two ways. For parameter
    ``{parameter}`` and mode ``{mode}`` you can either provide
    ``mod_{mode}_{parameter}`` or ``fdiff_{mode}_{parameter}``. If the former,
    the given value will be used for the parameter for the given mode. If the
    latter, the parameter will be scaled by 1 + the given value for the
    specified mode.

    Special combinations of parameters are recognized. They are:

    * ``mchirp``, ``eta`` : if either of these are modified, then ``mass1``
      and ``mass2`` wil be adjusted accordingly. If the resulting ``eta``
      is unphysical (not in [0, 0.25]) or chirp mass (not > 0) is unphysical,
      a ``NoWaveformError`` is raised.
    * ``chi_eff``, ``chi_a``: if either of these are modified, then ``spin1z``
      and ``spin2z`` will be adjusted accordingly, using the modified ``mass1``
      and ``mass2``.
    * ``spin1_perp``, ``spin1_azimuthal`` : if either of these are modified,
      then ``spin1x`` and ``spin1y`` will be adjusted accordingly.
    * ``spin2_perp``, ``spin2_azimuthal``: same as above, but for the second
      object.

    If the resulting modified spins yield a magnitude > 1, a ``NoWaveformError``
    is raised.
      

    Parameters
    ----------
    base_approximant : str
        The waveform approximant to use.
    mode_array : array of tuples
        The modes to generate, e.g., ``[(2, 2), (3, 3)]``.
    mod_{mode}_{parameter} : float, optional
        Use the given value for ``{parameter}`` for mode ``{mode}``.
    fdiff_{mode}_{parameter} : float, optional
        Adjust the parameter ``{parameter}`` for mode ``{mode}`` by the
        given fractional difference.
    other kwargs :
        All other keyword argument are passed to
        :py:func:`pycbc.waveform.waveform.get_fd_waveform`.

    Returns
    -------
    hp : FrequencySeries
        The plus polarization.
    hc : FrequencySeries
        The cross polarization.
    """
    # pull out the base wavefrom
    try:
        apprx = kwargs.pop("base_approximant")
    except KeyError:
        raise ValueError("Must provide a base_approximant")
    try:
        mode_array = kwargs.pop("mode_array")
    except KeyError:
        raise ValueError("Must provide a mode_array")
    # ensure mode array is a list of modes
    if isinstance(mode_array, str):
        mode_array = [tuple(int(m) for m in mode)
                      for mode in shlex.split(mode_array)]
    # set the approximant to the base
    kwargs["approximant"] = apprx
    # add default values, check for other required values
    kwargs = props(None, **kwargs)
    # pull out the modification arguments
    modargs = [p for p in kwargs
               if p.startswith("fdiff_") or p.startswith("absdiff_")
               or p.startswith("replace_")]
    modargs = dict((p, kwargs.pop(p)) for p in modargs)
    # parse the parameters
    modparams = {}
    for p, val in modargs.items():
        # template is (fdiff|mod)_mode_param
        diffarg, mode, param = p.split('_', 2)
        mode = tuple(int(x) for x in mode)
        try:
            addto = modparams[mode]
        except KeyError:
            addto = modparams[mode] = {}
        if param in addto:
            # the parameter is already there; means multiple args were given
            raise ValueError("Provide only one of absdiff_{m}_{p}, "
                             "fdiff_{m}_{p}, or replace_{m}_{p}".format(
                             m=''.join(map(str, mode)), p=param))
        addto[param] = (val, diffarg)
    # cycle over the modes, generating the waveform one at a time
    hps = []
    hcs = []
    wfargs = kwargs.copy()
    size = 0
    for mode in mode_array:
        # make sure mode is a tuple
        mode = tuple(mode)
        if mode in modparams:
            # convert mchirp, eta to mass1, mass2 if they are provided
            mchirp_mod = modparams[mode].pop('mchirp', None)
            eta_mod = modparams[mode].pop('eta', None)
            if mchirp_mod is not None or eta_mod is not None:
                m1, m2 = transform_masses(kwargs['mass1'], kwargs['mass2'],
                                          mchirp_mod, eta_mod)
                wfargs['mass1'] = m1
                wfargs['mass2'] = m2
            # convert spins
            chieff_mod = modparams[mode].pop('chi_eff', None)
            chia_mod = modparams[mode].pop('chi_a', None)
            if chieff_mod is not None or chia_mod is not None:
                s1z, s2z = transform_spinzs(wfargs['mass1'], wfargs['mass2'],
                                            kwargs['spin1z'], kwargs['spin2z'],
                                            chieff_mod, chia_mod)
                wfargs['spin1z'] = s1z
                wfargs['spin2z'] = s2z
            spin1_perp_mod = modparams[mode].pop('spin1_perp', None)
            spin1_az_mod = modparams[mode].pop('spin1_azimuthal', None)
            if spin1_perp_mod is not None or spin1_az_mod is not None:
                s1x, s1y = transform_spin_perp(kwargs['spin1x'],
                                               kwargs['spin1y'],
                                               spin1_perp_mod, spin1_az_mod)
                wfargs['spin1x'] = s1x
                wfargs['spin1y'] = s1y
            spin2_perp_mod = modparams[mode].pop('spin2_perp', None)
            spin2_az_mod = modparams[mode].pop('spin2_azimuthal', None)
            if spin2_perp_mod is not None or spin2_az_mod is not None:
                s2x, s2y = transform_spin_perp(kwargs['spin2x'],
                                               kwargs['spin2y'],
                                               spin2_perp_mod, spin2_az_mod)
                wfargs['spin2x'] = s2x
                wfargs['spin2y'] = s2y
            # update all other parameters
            for p in list(modparams[mode].keys()):
                diff, modtype = modparams[mode].pop(p)
                wfargs[p] = apply_mod(kwargs[p], diff, modtype)
            # check that we still have physical spins
            for obj in [1, 2]:
                mag = (wfargs['spin{}x'.format(obj)]**2 +
                       wfargs['spin{}y'.format(obj)]**2 +
                       wfargs['spin{}z'.format(obj)]**2)**0.5
                if mag >= 1:
                    raise NoWaveformError("unphysical spins")
        wfargs['mode_array'] = [mode]
        hp, hc = get_fd_waveform(**wfargs) 
        hps.append(hp)
        hcs.append(hc)
        size = max(size, len(hp))
    # make sure everything is the same size
    for ii, hp in enumerate(hps):
        hp.resize(size)
        hcs[ii].resize(size)
    return sum(hps), sum(hcs)


def apply_mod(origval, diff, modtype):
    """Applies modification to a parameter value.i
    
    Parameters
    ----------
    origval : float
        The original parameter value.
    diff : float
        The modification value.
    modtype : {'fdiff', 'absdiff', 'replace'}
        How to modify the original value.

    Returns
    -------
    float :
        The modified parameter value.
    """
    if modtype == 'fdiff':
        newval = origval * (1 + diff)
    elif modtype == 'absdiff':
        newval = origval + diff
    elif modtype == 'replace':
        newval = diff
    else:
        raise ValueError("unrecognized modtype {}".format(modtype))
    return newval


def transform_masses(mass1, mass2, mchirp_mod, eta_mod):
    """Modifies masses given a difference in mchirp and eta.

    Parameters
    ----------
    mass1 : float
        Mass of the larger object (already modified).
    mass2 : float
        Mass of the smaller object (already modified).
    mchirp_mod : tuple of (float, str) or None
        Tuple giving the modification value for ``mchirp``, and a string
        indicating whether the given modification is a fractional difference
        (``'fdiff'``), an absolute difference (``'absdiff'``), or a replacement
        (``'replace'``). If ``None``, the chirp mass will not be modified.
    eta_mod : tuple of (float, str)
        Same as ``mchirp_mod``, but for ``eta``.

    Returns
    -------
    mass1 : float
        Modified mass1.
    mass2 : float
        Modified mass2.
    """
    mchirp = conversions.mchirp_from_mass1_mass2(mass1, mass2)
    eta = conversions.eta_from_mass1_mass2(mass1, mass2)
    if mchirp_mod is not None:
        diff, modtype = mchirp_mod
        mchirp = apply_mod(mchirp, diff, modtype)
    if eta_mod is not None:
        diff, modtype = eta_mod
        eta = apply_mod(eta, diff, modtype)
    # make sure values are physical
    if (eta < 0 or eta > 0.25) or mchirp < 0:
        raise NoWaveformError("unphysical masses")
    m1 = conversions.mass1_from_mchirp_eta(mchirp, eta)
    m2 = conversions.mass2_from_mchirp_eta(mchirp, eta)
    return m1, m2


def transform_spinzs(mass1, mass2, spin1z, spin2z, chieff_mod, chia_mod):
    """Modifies spinzs given a difference in chieff and chia.

    Raises a ``NoWaveformError`` if the modified ``chi_eff``, ``chi_a``,
    ``spin1z``, or ``spin2z`` are not in (-1, 1).

    Parameters
    ----------
    mass1 : float
        Mass of the larger object (already modified).
    mass2 : float
        Mass of the smaller object (already modified).
    spin1z : float
        Z-component of spin of object 1 to modify.
    spin2z : float
        Z-component of spin of object 2 to modify.
    chieff_mod : tuple of (float, str) or None
        Tuple giving the modification value for ``chi_eff``, and a string
        indicating whether the given modification is a fractional difference
        (``'fdiff'``), an absolute difference (``'absdiff'``), or a replacement
        (``'replace'``). If ``None``, ``chi_eff`` will not be modified.
    chia_mod : tuple of (float, str) or None
        Same as ``chieff_mod``, but for ``chi_a``.

    Returns
    -------
    spin1z : float
        Modified spin1z
    spin2z : float
        Modified spin2z
    """
    chi_eff = conversions.chi_eff(mass1, mass2, spin1z, spin2z)
    chi_a = conversions.chi_a(mass1, mass2, spin1z, spin2z)
    if chieff_mod is not None:
        diff, modtype = chieff_mod
        chi_eff = apply_mod(chi_eff, diff, modtype)
        if abs(chi_eff) >= 1:
            raise NoWaveformError("unphysical chi_eff")
    if chia_mod is not None:
        diff, modtype = chia_mod
        chi_a = apply_mod(chi_a, diff, modtype)
        if abs(chi_a) >= 1:
            raise NoWaveformError("unphysical chi_a")
    spin1z = conversions.spin1z_from_mass1_mass2_chi_eff_chi_a(mass1, mass2,
                                                               chi_eff, chi_a)
    if abs(spin1z)>= 1:
        raise NoWaveformError("unphysical spin1z")
    spin2z = conversions.spin2z_from_mass1_mass2_chi_eff_chi_a(mass1, mass2,
                                                               chi_eff, chi_a)
    if abs(spin2z) >= 1:
        raise NoWaveformError("unphysical spin2z")
    return spin1z, spin2z


def transform_spin_perp(spinx, spiny, spin_perp_mod, spin_azimuthal_mod):
    """Modifies x and y components of spin of an object.

    Raises a ``NoWaveformError`` if the modified spin perp magnitude is not
    in (-1, 1). Also raises a ``ValueError`` if absolute spin azimuthal
    modification is not in `[-pi/2, pi/2]`.

    Parameters
    ----------
    spinx : float
        X-component of spin to modify.
    spiny : float
        Y-component of spin to modify.
    spin_perp_mod : tuple of (float, str) or None
        Tuple giving the modification value for ``spin_perp``, and a string
        indicating whether the given modification is a fractional difference
        (``'fdiff'``), an absolute difference (``'absdiff'``), or a replacement
        (``'replace'``). If ``None``, ``spin_perp`` will not be modified.
    spin_azimuthal_mod : tuple of (float, str) or None
        Same as ``spin_perp_mod``, but for ``spin_azimuthal``.

    Returns
    -------
    spinx : float
        Modified spinx
    spiny : float
        Modified spiny
    """
    spin_az = numpy.arctan2(spiny, spinx)
    spin_perp = (spinx**2 + spiny**2)**0.5
    if spin_perp_mod is not None:
        diff, modtype = spin_perp_mod
        spin_perp = apply_mod(spin_perp, diff, modtype)
        if abs(spin_perp) >= 1:
            raise NoWaveformError("unphysical spin perp")
    if spin_azimuthal_mod is not None:
        diff, modtype = spin_azimuthal_mod
        # make sure absolute differences are +/- pi/2
        if diff == 'absdiff' and abs(diff) > numpy.pi/2:
            raise ValueError("absolute spin azimuthal difference must be in "
                             "[-pi/2, pi/2]")
        spin_az = apply_mod(spin_az, diff, modtype)
    # if spin perp is now negative, add pi to the azimuthal
    if spin_perp < 0:
        spin_az += numpy.pi
        spin_perp = abs(spin_perp)
    # convert back to x, y
    spinx = spin_perp * numpy.cos(spin_az)
    spiny = spin_perp * numpy.sin(spin_az)
    return spinx, spiny
