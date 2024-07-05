# Rough estimate for the diffuse gamma-ray emission

**Step 1** Clone the repository and download the dust maps from [Edenhofer et al. A&A 2023](https://zenodo.org/records/10658339)
```sh
wget https://zenodo.org/records/10658339/files/samples_healpix.fits?download=1
```

**Step 2** Make sure the sample dust maps has name 'samples_healpix.fits' then run 0_dust_map.py to extract gas column density maps (a dust-to-gas ratio is assumed)
```sh
python3 0_dust_map.py
```

**Step 3** Run 1_gamma_map.py to get the gamma maps from different gas maps (note that dust data extend to only 1.25 kpc so this is suitable only for high lattitude). Note also that the gamma-ray maps are derived using the local cosmic-ray spectra fitted by [Boschini et al. ApJS 2020](https://ui.adsabs.harvard.edu/abs/2020ApJS..250...27B/abstract)
```sh
python3 1_gamma_map.py
```

**Step 4** Plot some illutration
```sh
python3 2_plot.py
```
