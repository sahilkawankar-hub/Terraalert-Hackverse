# Data Sources — TerraAlert Ingest

Ingestion timestamp: 2026-10-06 04:52:10 UTC
Event: Assam Floods, June 2022 (Barpeta-Nalbari, Brahmaputra north bank)
AOI: (90.95, 26.1, 91.25, 26.37) (EPSG:4326)
Grid: EPSG:32646 at 20m resolution

### 1. Copernicus Sentinel-1 Synthetic Aperture Radar (SAR)
- **Collection**: `COPERNICUS/S1_GRD`
- **Instrument Mode**: Interferometric Wide Swath (IW)
- **Polarisation**: VV
- **Orbit Pass**: DESCENDING, Relative Orbit 150
- **Pre-event window**: 2022-04-20 to 2022-05-10 (acquisition: 2022-05-06)
- **Post-event window**: 2022-06-17 to 2022-06-30 (acquisition: 2022-06-23)
- **Files**: `outputs/pre.tif`, `outputs/post.tif` (dB backscatter, nodata -9999)
- **Attribution**: Contains modified Copernicus Sentinel data [2022], processed by ESA and Google Earth Engine.

### 2. JRC Global Surface Water (GSW v1.4)
- **Asset**: `JRC/GSW1_4/GlobalSurfaceWater`
- **Band**: `occurrence` (> 50% = permanent water)
- **File**: `outputs/perm_water.tif` (binary 0/1, nodata 255)
- **Attribution**: European Commission Joint Research Centre (JRC).

### 3. NASA Shuttle Radar Topography Mission (SRTM)
- **Asset**: `USGS/SRTMGL1_003` (1 arc-second, ~30m)
- **Bands**: `elevation` (meters), `slope` (derived in degrees via ee.Terrain.slope)
- **Files**: `outputs/dem.tif`, `outputs/slope.tif` (nodata -9999)
- **Attribution**: NASA / USGS.

### 4. WorldPop Unconstrained Individual Countries (2020)
- **Asset**: `WorldPop/GP/100m/pop` (IND, 2020)
- **Band**: `population` (people per pixel, count-preserving volume resampled to 20m)
- **File**: `outputs/pop.tif` (nodata -9999)
- **Attribution**: WorldPop (www.worldpop.org - School of Geography and Environmental Science, University of Southampton).
