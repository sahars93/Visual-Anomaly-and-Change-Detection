
# Visual anomaly and change detection

Analysis of  Self-Calibrating Anomaly and Change Detection for Autonomous Inspection Robots

Full paper PDF: [Self-Calibrating Anomaly and Change Detection for Autonomous Inspection Robots](https://arxiv.org/pdf/2209.02379.pdf).

## Introduction

This work proposes a comprehensive deep learning framework for detecting anomalies and changes in a priori unknown environments after a reference dataset is gathered. We use the SuperPoint and SuperGlue feature extraction and matching and instance segmentation methods to detect anomalies based on reference images from a similar location. 

## How to run

   0. Clone repo
      ```
      git clone git@github.com:sahars93/Visual-Anomaly-and-Change-Dtecection-.git
      ```
   1. Install Detectron2 and Superpoint:
      ```
      cd Visual-Anomaly-and-Change-Dtecection-/
      git submodule update --init --recursive
      ```
   2. Run
      ```
      python3 final_anomaly_detection.py

## Self-Calibration
<div align=center>
  <img src="./self_calibration/calibration_results/cameras_key_thresh0.003.png" width="500" />
  <p align="center">Calubration results for three different cameras</p>
</div>



## Results
![](./output_images/all_together.png)
