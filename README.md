## Rough estimate for the diffuse gamma-ray emission

**Step 1** Clone the repository and download the dust maps from [Söding et al. 2024](https://zenodo.org/records/12578443)
```sh
wget https://zenodo.org/records/12578443/files/samples_densities_hpixr.fits?download=1
```

**Step 2** Make sure the sample gas maps has name 'samples_densities_hpixr.fits' then run 0_gas_map.py to see mean column density maps for HI and H2
```sh
python3 0_gas_map.py
```

**Step 3** Run 1_gamma_map.py to get the gamma maps from different gas maps. Note also that the gamma-ray maps are derived using the local cosmic-ray spectra fitted by [Boschini et al. ApJS 2020](https://ui.adsabs.harvard.edu/abs/2020ApJS..250...27B/abstract)
```sh
python3 1_gamma_map.py
```

**Step 4** Plot some illutration
```sh
python3 2_plot.py
```
