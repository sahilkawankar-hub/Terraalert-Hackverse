# TerraAlert — Data Sources & Attributions

This document registers all primary and ancillary geospatial datasets utilized across the TerraAlert flood change detection and impact analysis pipeline, along with citations, access endpoints, licenses, and preprocessing methodologies.

---

## 1. Remote Sensing Datasets

### 1.1 Copernicus Sentinel-1 Synthetic Aperture Radar (SAR)
- **Sensor:** C-band Synthetic Aperture Radar (SAR) onboard Sentinel-1A and Sentinel-1B.
- **Instrument Mode:** Interferometric Wide Swath (IW), Ground Range Detected (GRD) at 10 m spatial resolution.
- **Polarizations:** VV (primary co-polarization for specular water detection) and VH (cross-polarization for volume scattering).
- **Provider:** European Space Agency (ESA) / European Commission Copernicus Programme.
- **Collection Access:** Google Earth Engine (`COPERNICUS/S1_GRD_FLOAT`) or Copernicus Data Space Ecosystem.
- **License:** [Copernicus Open Access Policy](https://sentinels.copernicus.eu/web/sentinel/sentinel-data-access/rights-and-licensing) (Free, full, and open access).
- **Processing:** Multi-temporal speckle filtering, thermal noise removal, radiometric calibration to $\sigma^0$ backscatter (decibels), and terrain correction via SRTM 30 m DEM.

### 1.2 Copernicus Sentinel-2 Multi-Spectral Instrument (MSI)
- **Sensor:** Multi-Spectral Instrument (MSI) optical imagery (Level-2A Bottom-Of-Atmosphere reflectance).
- **Provider:** European Space Agency (ESA) / Copernicus Programme.
- **Collection Access:** Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`).
- **Bands Used:** Green (B3, 560 nm) and NIR (B8, 842 nm) for Normalized Difference Water Index (NDWI), plus SCL scene classification for cloud/shadow masking.
- **License:** Copernicus Open Access Policy.
- **Application:** Multi-sensor verification and optical cross-validation during cloud-free post-event windows.

---

## 2. Ancillary & Topographic Datasets

### 2.1 NASA Shuttle Radar Topography Mission (SRTM) DEM
- **Product:** SRTM Version 3.0 Global 1 arc-second (~30 m resolution).
- **Provider:** NASA Jet Propulsion Laboratory (JPL) / USGS.
- **Collection Access:** Google Earth Engine (`USGS/SRTMGL1_003`).
- **License:** Public Domain / NASA open data.
- **Application:** Terrain slope computation ($< 5^\circ$ mask) to eliminate radar shadow, layover artifacts, and false-positive water classifications on steep slopes.

### 2.2 JRC Global Surface Water (GSW) v1.4
- **Product:** Joint Research Centre Global Surface Water Mapping (1984–2021 history).
- **Authors:** Jean-François Pekel, Andrew Cottam, Noel Gorelick, Alan S. Belward (Nature 540, 418-422, 2016).
- **Collection Access:** Google Earth Engine (`JRC/GSW1_4/GlobalSurfaceWater`).
- **License:** European Commission Open Access.
- **Application:** Identification and masking of permanent/semi-permanent water bodies (water occurrence $> 80\%$) to isolate true novel flood events from historical baseline hydrology.

---

## 3. Human Exposure & Infrastructure Datasets

### 3.1 WorldPop Global High Resolution Population Counts
- **Product:** Constrained UN-Adjusted 100 m gridded population counts (2020).
- **Provider:** WorldPop, School of Geography and Environmental Science, University of Southampton.
- **Citation:** WorldPop (www.worldpop.org - School of Geography and Environmental Science, University of Southampton). 2020. Global 100m Population Counts.
- **License:** Creative Commons Attribution 4.0 International ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Application:** Estimation of exposed populations in priority response grid cells via count-preserving conservative resampling.

### 3.2 OpenStreetMap (OSM) Road Network & Critical Amenities
- **Product:** High-density highway vectors (trunk, primary, secondary, tertiary, residential) and critical infrastructure points/polygons (hospitals, clinics, schools, fire stations, police stations, government offices).
- **Source:** © OpenStreetMap contributors via Overpass API / OSMnx.
- **License:** Open Database License 1.0 ([ODbL 1.0](https://opendatacommons.org/licenses/odbl/)).
- **Application:** Calculation of inundated road segment lengths (cut roads) and flooded / isolated emergency infrastructure to compute compounding priority rankings.

---

## 4. Machine Learning & Reference Datasets

### 4.1 Sen1Floods11 Ground Truth Dataset
- **Description:** Georeferenced dataset for flood detection containing 4,831 chips of co-registered Sentinel-1 SAR and Sentinel-2 optical imagery across 11 global flood events, with expert quality-controlled hand-labeled flood masks.
- **Citation:** Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, V. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. *CVPR Workshops 2020*, 210-211.
- **License:** Creative Commons Attribution 4.0 International ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Application:** Offline supervised benchmarking and validation of change detection accuracy (Precision, Recall, F1, IoU).

### 4.2 Prithvi EO Foundation Models
- **Provider:** NASA / IBM Research (Hugging Face / GitHub).
- **License:** Apache License 2.0.
- **Application:** Geospatial foundation model backbone for multi-spectral segmentation and feature extraction.
