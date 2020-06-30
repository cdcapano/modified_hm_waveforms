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

from pycbc import conversions
from pycbc.waveform.waveform import (get_fd_waveform, props)


def modhm_fd(**kwargs):
    """Allows a waveform to be generated with different parameters for the
    sub-dominant modes.
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
    # set the approximant to the base
    kwargs["approximant"] = apprx
    # add default values, check for other required values
    kwargs = props(None, **kwargs)
    # pull out the fractional arguments
    modargs = [p for p in kwargs if p.startswith("fdiff_")]
    modargs = dict((p, kwargs.pop(p)) for p in modargs)
    # parse the parameters
    modparams = {}
    for p, val in modargs.items():
        # template is fdiff_mode_param
        _, mode, param = p.split('_', 2)
        mode = tuple(int(x) for x in mode)
        try:
            addto = modparams[mode]
        except KeyError:
            addto = modparams[mode] = {}
        addto[param] = val
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
            mchirp_fdiff = modparams[mode].pop('mchirp', 0.)
            eta_fdiff = modparams[mode].pop('eta', 0.)
            if mchirp_fdiff or eta_fdiff:
                m1 = kwargs['mass1']
                m2 = kwargs['mass2']
                mchirp = conversions.mchirp_from_mass1_mass2(m1, m2)
                eta = conversions.eta_from_mass1_mass2(m1, m2)
                # scale
                mchirp *= 1 + mchirp_fdiff
                eta *= 1 + eta_fdiff
                # make sure values are physical
                eta = max(min(eta, 0.25), 0)
                m1 = conversions.mass1_from_mchirp_eta(mchirp, eta)
                m2 = conversions.mass2_from_mchirp_eta(mchirp, eta)
                wfargs['mass1'] = m1
                wfargs['mass2'] = m2
            # update all other parameters
            for p in list(modparams[mode].keys()):
                fdiff = modparams[mode].pop(p)
                wfargs[p] = kwargs[p] * (1 + fdiff)
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
