### Clone the repository and download the dust maps from Edenhofer et al. A&A 2023 (from the following site https://zenodo.org/records/10658339) as follows
wget https://zenodo.org/records/10658339/files/samples_healpix.fits?download=1

### Make sure the sample dust maps has name 'samples_healpix.fits' then run 0_dust_map.py to extract gas column density maps (a dust-to-gas ratio is assumed)
python3 0_dust_map.py

### Run 1_gamma_map.py to get the gamma maps from different gas maps (note that dust data extend to only 1.25 kpc so this is suitable only for high lattitude). Note 
### also that the gamma-ray maps are derived using the local cosmic-ray spectra fitted by Boschini et al. ApJS 2020
python3 1_gamma_map.py

### Plot some illutration
python3 2_plot.py
