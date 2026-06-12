# StacksOnStacks

Data information:
[Drive with data & other files.](https://drive.google.com/drive/folders/1eAorLCxPoz7HXJv8tv4DCH6_aXtvx-K_?usp=sharing)
* ***90.150_overlap_BgsLrgDuplicates.fit***: Table with applicable physical properties, redshift, RA, Dec, Galaxy Type, and tSZ profiles (using the CAP filter ringring2) measured in the 98 and 150 GHz regime. These profiles were run using a 98 or 150 GHz map, and mask combining a cluster mask and footprint mask. All galaxies in this catalog are those that exist in DESI DR1 and the map / mask overlap, with no additional cuts.
* ***90.150.kappa_overlap_BgsLrgDuplicates.fits***: Table with applicable physical properties, redshift, RA, Dec, Galaxy Type, tSZ profiles (using the CAP filter ringring2) measured in the 98 and 150 GHz regime, and lensing profiles "kappa" (using the CAP filter ring). The tSZ profiles were run using a 98 or 150 GHz map, the lensing profile using the ACT lensing map, and mask combining a cluster mask and footprint mask. All galaxies in this catalog are those that exist in DESI DR1 and the map / mask overlap, with no additional cuts.
    * It is important to note that both of these tables have some objects duplicated, some indicated as BGS and some indicated as LRG. To make a new table and keep only one version of every duplicate, please see otherFun.resolve_duplicates

We should add separate python scripts for:
- ~~boxing (Abby) done~~
- ~~virial radii (Abby) done~~
- ~~covariance estimation (Abby)~~
- theory/fitting codes (Noah)
