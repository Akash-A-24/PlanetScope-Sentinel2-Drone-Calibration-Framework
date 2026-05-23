# PlanetScope-Sentinel2-Drone-Calibration-Framework

## Overview

This repository provides a complete machine learning-based remote sensing calibration workflow for integrating:

- UAV multispectral imagery
- Sentinel-2 satellite imagery
- PlanetScope SuperDove imagery

The framework calibrates PlanetScope reflectance data using high-resolution UAV observations and Sentinel-2 spectral information through supervised regression models.

The workflow is designed for:

- Precision agriculture
- Crop stress monitoring
- Multi-sensor data harmonization
- Vegetation index generation
- Reflectance calibration studies
- High-resolution agricultural monitoring

---

# Features

## Multi-Sensor Integration
- UAV multispectral imagery
- Sentinel-2 Level-2A imagery
- PlanetScope Surface Reflectance imagery

## Machine Learning Calibration
Supports:
- Random Forest Regression
- Gradient Boosting Regression

## Automated Processing
- PlanetScope ZIP extraction
- Spatial reprojection and resampling
- Pixel-wise training sample generation
- Reflectance calibration
- Vegetation index generation

## Vegetation Indices
Automatically generates:
- NDVI
- NDRE
- GNDVI
- RVI

## Output Products
- Calibrated PlanetScope bands
- Calibrated vegetation indices
- Flat CSV/NumPy outputs
- Saved ML models and scalers

---

# Workflow

```text
Drone Multispectral Data
          +
Sentinel-2 Imagery
          +
PlanetScope Imagery
          ↓
Spatial Resampling
          ↓
Training Dataset Creation
          ↓
Machine Learning Calibration
          ↓
Calibrated Reflectance Outputs
          ↓
Vegetation Index Generation
