## Rough estimate for the diffuse gamma-ray emission

We provide a rough estimate for the diffuse gamma-ray emission from cosmic-ray hadrons. Two ingredients are required: i) Galactic cosmic-ray propagation model and ii) Galactic gas distribution.

**Cosmic-ray propagation model**: We use the axisymmetric model for Galactic cosmic-ray distribution from [Maurin et al. 2001](https://ui.adsabs.harvard.edu/abs/2001ApJ...555..585M/abstract). The distribution of cosmic-ray source and the diffusion coefficient are respectively from [Yusifov et al. 2004](https://ui.adsabs.harvard.edu/abs/2004A%26A...422..545Y/abstract) and [Genolini et al. 2017](https://ui.adsabs.harvard.edu/abs/2017PhRvL.119x1101G/abstract). We estimate the gamma-ray emissivity using cross sections from [Kafexhiu et al. 2014](https://ui.adsabs.harvard.edu/abs/2014PhRvD..90l3014K/abstract). We use directly the associated library [LibppGam](https://github.com/ervinkafex/LibppGam/blob/main/Python/LibppGam.py) from which 4 sets of parametrizations for cross-sections as simulated from Geant 4, Pythia, SIBYLL, and QGSJET can be chosen for this estimate. 

**Galactic gas distribution**: Gas distribution is taken from [Söding et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240702859S/abstract) inferred from emission lines using [NIFTy (Frank et al. 2021)](https://ui.adsabs.harvard.edu/abs/2021Entrp..23..853F/abstract). They use the 21-cm line atomic hydrogen and the J=1&rarr;0 line of <sup>12</sup>CO which allow to trace density of both atomic and molecular hydrogen.         

**Instruction to use the code**: 
Clone the repository and download the gas maps from [Söding et al. 2024](https://zenodo.org/records/12578443) as follows
```sh
wget https://zenodo.org/records/12578443/files/samples_densities_hpixr.fits?download=1
```

Make sure the sample gas maps has name 'samples_densities_hpixr.fits'. This file contains 8 samples of atomic and molecular hydrogen density. You can run '0_gas_map.py' to see mean column density maps for each gas species. You can then run '1_CR_map.py' to get a file containing gamma-ray maps at different energy (some other intermediate results like surface density of cosmic-ray sources, local cosmic-ray spectrum, and cosmic-ray distribution on Galactic scales are also shown).
```sh
python3 0_gas_map.py
python3 1_CR_map.py
```
Finally, plot gamma-ray map and the correspodning uncertainty due to gas using '2_plot.py'
```sh
python3 2_plot.py
```

You will find the gamma-ray maps as shown below.

![Diffuse gamma-ray map and uncertainty](https://drive.google.com/file/d/1nxmu7CNI7E_eezKqspYkMoJtsizDeE9s)
