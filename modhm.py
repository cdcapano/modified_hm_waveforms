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
from pycbc.waveform.waveform import (get_fd_waveform, props)


def modhm_fd(**kwargs):
    r"""Allows a waveform to be generated with different parameters for the
    sub-dominant modes.

    Parameters
    ----------
    base_approximant : str
        The waveform approximant to use.
    mode_array : array of tuples
        The modes to generate, e.g., ``[(2, 2), (3, 3)]``.
    fdiff_{mode}_{parameter} : float, optional
        Adjust the parameter ``{parameter}`` for mode ``{mode}`` by the
        given fractional difference. The ``{parameter}`` must be one of the
        other keyword arguments provide, or ``mchirp`` or ``eta``. If the
        later, the ``mass1`` and ``mass2`` values will be modified accordingly.
        For example, ``fdiff_33_mchirp = -0.1`` will make the chirp mass used
        to generate the 33 mode be 10% smaller than the mchirp specified
        by the given ``mass1`` and ``mass2`` parameters (and used for all
        other modes aside from the 33 mode).
    absdiff_{mode}_{parameter} : float, optional
        Same as ``fdiff_{mode}_{parameter}``, but apply an absolute difference
        to the parameter. For example, ``absdiff_33_coa_phase = 2`` would
        shift the ``coa_phase`` passed to the 33 mode by 2 radians with
        respect to the ``coa_phase`` passed to all other modes.
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
               if p.startswith("fdiff_") or p.startswith("absdiff_")]
    modargs = dict((p, kwargs.pop(p)) for p in modargs)
    # parse the parameters
    modparams = {}
    for p, val in modargs.items():
        # template is fdiff_mode_param
        diffarg, mode, param = p.split('_', 2)
        mode = tuple(int(x) for x in mode)
        try:
            addto = modparams[mode]
        except KeyError:
            addto = modparams[mode] = {}
        isabsdiff = diffarg.startswith("abs")
        if param in addto:
            # the parameter is already there; means both diff and fdiff were
            # specified
            raise ValueError("Both a fractional difference (starts with "
                             "'fdiff_') and an absolute difference (starts "
                             "with 'absdiff_') specified for parameter {} and "
                             "mode {}. Please only provide one or the other."
                             .format(param, ''.join(map(str, mode))))
        addto[param] = (val, isabsdiff)
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
            mchirp_diff, mcisabs = modparams[mode].pop('mchirp', (0, True))
            eta_diff, etaisabs = modparams[mode].pop('eta', (0, True))
            if mchirp_diff or eta_diff:
                m1 = kwargs['mass1']
                m2 = kwargs['mass2']
                mchirp = conversions.mchirp_from_mass1_mass2(m1, m2)
                eta = conversions.eta_from_mass1_mass2(m1, m2)
                # scale
                if mcisabs:
                    mchirp += mchirp_diff
                else:
                    mchirp *= 1 + mchirp_diff
                if etaisabs:
                    eta += eta_diff
                else:
                    eta *= 1 + eta_diff
                # make sure values are physical
                eta = max(min(eta, 0.25), 0)
                m1 = conversions.mass1_from_mchirp_eta(mchirp, eta)
                m2 = conversions.mass2_from_mchirp_eta(mchirp, eta)
                wfargs['mass1'] = m1
                wfargs['mass2'] = m2
            # update all other parameters
            for p in list(modparams[mode].keys()):
                diff, isabsdiff = modparams[mode].pop(p)
                if isabsdiff:
                    modp = kwargs[p] + diff
                else:
                    modp = kwargs[p] * (1 + diff)
                wfargs[p] = modp
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
