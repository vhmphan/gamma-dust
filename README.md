## Rough estimate for the diffuse gamma-ray emission

We provide a rough estimate for the diffuse gamma-ray emission from cosmic-ray hadrons. Two ingredients are required: i) Galactic cosmic-ray propagation model and ii) Galactic gas distribution.

**Cosmic-ray propagation model**: We use the axisymmetric model for cosmic-ray distribution from [Maurin et al. 2001](https://ui.adsabs.harvard.edu/abs/2001ApJ...555..585M/abstract). The distribution of cosmic-ray source and the diffusion coefficient are from [Yusikov et al. 2004](https://ui.adsabs.harvard.edu/abs/2004A%26A...422..545Y/abstract) and [Genolini et al. 2017](https://ui.adsabs.harvard.edu/abs/2017PhRvL.119x1101G/abstract). We estimate the gamma-ray emissivity using cross sections from [Kafexhiu et al. 2014](https://ui.adsabs.harvard.edu/abs/2014PhRvD..90l3014K/abstract). 

**Galactic gas distribution**: Gas distribution is taken from [Söding et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240702859S/abstract) inferred from emission lines using [NIFTy (Frank et al. 2021)](https://ui.adsabs.harvard.edu/abs/2021Entrp..23..853F/abstract). They use the 21-cm line atomic hydrogen and the J=1 &rarr; 0 line of <sup>12</sup>CO which allows to trace density of both atomic and molecular hydrogen.         

**Instruction to use the code**: 
Clone the repository and download the dust maps from [Söding et al. 2024](https://zenodo.org/records/12578443)
```sh
wget https://zenodo.org/records/12578443/files/samples_densities_hpixr.fits?download=1
```

Make sure the sample gas maps has name 'samples_densities_hpixr.fits' then run '0_gas_map.py' to see mean column density maps for atomic and molecular hydrogen and '1_CR_map.py' to get the gamma-ray map at 10 GeV (some other intermediate results like surface density of cosmic-ray sources and local cosmic-ray spectrum are also shown)
```sh
python3 0_gas_map.py
python3 1_CR_map.py
```
