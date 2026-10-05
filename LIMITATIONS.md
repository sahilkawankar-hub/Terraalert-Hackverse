# TerraAlert — Physical Limitations, Methodological Assumptions & Caveats

This document describes the physical constraints of remote sensing sensors, methodological assumptions, dataset uncertainties, and ethical considerations inherent to the TerraAlert flood intelligence platform.

Emergency managers, first responders, and decision-makers must account for these factors when interpreting system alerts.

---

## 1. Physical Constraints of Synthetic Aperture Radar (SAR)

Synthetic Aperture Radar operates by transmitting microwave pulses (C-band, 5.405 GHz, $\lambda \approx 5.6\text{ cm}$) and measuring backscatter intensity $\sigma^0$ reflected back to the sensor. While SAR penetrates clouds, fog, and operates day and night, it exhibits several well-documented physical failure modes in post-disaster scenarios:

### 1.1 The "Single-Image Trap" vs. Multi-Temporal Change Detection
- **The Phenomenon:** Permanent calm water bodies (lakes, reservoirs, wide rivers) exhibit low backscatter ($\approx -18\text{ dB}$ to $-24\text{ dB}$) due to specular reflection. Similarly, dry, smooth man-made surfaces (airport runways, paved multi-lane expressways, parking lots) also reflect radar pulses away from the sensor, producing low backscatter values nearly indistinguishable from water in an isolated post-event acquisition.
- **TerraAlert Mitigation:** TerraAlert enforces **multi-temporal change detection** ($\Delta\sigma^0 = \sigma^0_{\text{post}} - \sigma^0_{\text{pre}} \le -3.0\text{ dB}$) combined with permanent water masking via JRC Global Surface Water. Pixels are classified as flooded *only* if they experienced a significant backscatter decline relative to the pre-event dry baseline.

### 1.2 Urban Double-Bounce and Building Occlusion
- **The Limitation:** In dense urban corridors, vertical building walls and horizontal street surfaces form dihedral right-angle corner reflectors. This geometry causes "double-bounce" scattering that redirects microwave energy directly back to the sensor, creating anomalously bright radar signals ($\sigma^0 > -8\text{ dB}$).
- **Impact:** Submerged streets flanked by multi-story masonry structures often reflect high backscatter rather than low specular backscatter, leading to **false negatives** (under-detection) in flooded urban cores.
- **Operational Guidance:** Ground reconnaissance and road network connectivity models must supplement SAR-derived urban flood masks.

### 1.3 Dense Vegetation Canopy Penetration
- **The Limitation:** Sentinel-1 C-band microwaves have a wavelength of approximately 5.6 cm, which scatters primarily within tree canopies, leaves, and dense foliage (volume scattering). C-band cannot reliably penetrate thick forest or dense agricultural canopies to detect standing water beneath the foliage.
- **Impact:** Flooded forests or submerged crops beneath mature canopies will register typical vegetation backscatter rather than water, causing under-detection. Longer wavelength systems (such as L-band SAR, e.g. NASA-ISRO NISAR, ALOS-2 PALSAR) are required for full sub-canopy inundation mapping.

### 1.4 Wind-Induced Surface Roughness
- **The Limitation:** Strong storm winds, tropical cyclones, or torrential downpours during satellite overpass agitate standing water surfaces, producing capillary waves. This surface roughness triggers Bragg scattering that increases backscatter above classical water thresholds.
- **Impact:** Wind-roughened open floodwaters may fail static decibel thresholding, causing temporary false negatives during stormy overpasses.

### 1.5 Radar Layover, Shadow, and Topographic Distortion
- **The Limitation:** In rugged terrain, side-looking radar geometry causes radar layover (steep slopes facing the radar appear compressed) and radar shadows (slopes facing away receive no signal and register near zero backscatter).
- **TerraAlert Mitigation:** TerraAlert applies a strict slope mask using the NASA SRTM 30 m DEM ($\text{slope} \le 5^\circ$). Steep hillsides where flash floods cannot pool as standing water are automatically excluded from false water classifications.

---

## 2. Temporal Resolution & Flood Crest Dynamics

### 2.1 Satellite Revisit Window
- **Constraint:** Sentinel-1 possesses a constellation revisit cycle of 6 to 12 days for a given orbital geometry.
- **Operational Reality:** Flash floods, riverine dam breaches, or tidal surges often reach peak crest within 6 to 24 hours and recede before an orbital pass. A satellite image acquired 3 days post-event records the *residual flood extent* at the precise instant of sensor exposure, not necessarily the historical maximum inundation.
- **Caveat:** The time gap between pre- and post-acquisitions must be considered when evaluating event severity.

---

## 3. Human Exposure & Infrastructure Assumptions

### 3.1 WorldPop Population Disaggregation
- **Assumption:** Population figures in TerraAlert are calculated by conservatively resampling 100 m gridded WorldPop counts onto the metric analysis grid and multiplying by the zonal flood area fraction:
  $$\text{people\_affected} = \text{population}_{\text{zone}} \times \left(\frac{\text{flood\_pct}}{100}\right)$$
- **Limitations:**
  1. *Uniform Distribution Assumption:* This formulation assumes population is distributed homogeneously across the grid cell. If an urban settlement occupies one corner and floodwater covers the uninhabited farmland corner, the estimate may overestimate exposure.
  2. *Ambient vs. Dynamic Population:* WorldPop models night-time residential distribution; it does not account for daytime work mobility, evacuation movements, or temporary relief shelter aggregations.
  3. *Statistical Nature:* All population counts are mathematical approximations for macroscopic resource staging, not verified on-ground headcounts.

### 3.2 OpenStreetMap Data Completeness
- **Limitation:** OpenStreetMap data is crowd-sourced and volunteer-curated. Completeness, attribute tagging fidelity, and geometric precision vary markedly between metropolitan centers and remote rural regions.
- **Impact:** Unmapped dirt access roads or uncatalogued rural primary health clinics will not contribute to road cut or facility vulnerability scores.

---

## 4. Confidence Heuristic & The `VERIFY` Tier

### 4.1 Heuristic Nature of Confidence Scores
- The confidence score (0.0 to 1.0) produced in TerraAlert is an engineering heuristic synthesized from:
  1. Multi-method agreement (classical threshold vs. ML segmentation).
  2. Topographic slope penalties.
  3. Temporal proximity between pre- and post-event imagery.
  4. Decision margin distance from threshold boundaries.
- **Notice:** It is not a formal Bayesian posterior probability or calibrated statistical uncertainty.

### 4.2 The `VERIFY` Protocol
- High-priority zones (top tiers P1 and P2) that exhibit low detection confidence (confidence score $< 0.40$ or multi-method disagreement) are automatically tagged with the **`VERIFY`** tier.
- **Mandate:** Emergency operational teams must **not** dispatch irreversible heavy assets (e.g. heavy earth-moving equipment, bridge repair teams) based solely on a `VERIFY` alert without prior drone inspection, helicopter flyover, or local ground scout confirmation.
