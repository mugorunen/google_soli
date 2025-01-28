# Google Soli - Radar Gesture Recognition

## Overview

Google Soli is a solid-state millimeter-wave radar developed by Google for dynamic gesture recognition applications. The Soli chip is compact and power-efficient, designed to prioritize temporal resolution over spatial resolution. This allows it to detect subtle, non-rigid motions with high precision.

### How It Works

Soli utilizes antennas that function as both transmitters and receivers. A signal is emitted from the transmitter antenna and reflected back to the receiver antenna. The raw signals received are then processed into **Range-Doppler images**, which are used for gesture classification. In these images:

- **Pixel intensity** represents the power of the received signals.
- The **horizontal axis** corresponds to the velocity of the movement.
- The **vertical axis** represents the distance of the action from the chip.

---

## Deep-Soli Dataset

The Deep-Soli dataset contains gesture data collected from 10 subjects performing 11 distinct gesture movements. Each subject performed 25 sessions per gesture, resulting in a total of **11 × 10 × 25 recorded sessions**.

### Dataset Structure

- The dataset is stored in `.h5` files, with each file named in the format:  
  `LabelNumber_SubjectNumber_SessionNumber.h5`.
- Each file contains **4 channels**, corresponding to data recorded simultaneously from 4 receiver antennas.
- Each channel consists of **1024 pixels**, representing **32 × 32 Range-Doppler images**.

---

## Tasks and Jupyter Notebooks

The repository includes **4 Jupyter notebooks** for different tasks:

### Task A
- **Objective**: Analyze the dataset, preprocess the data, and train a model to achieve the highest possible accuracy.

### Task B_1
- **Objective**: Adapt the network to run on a chip with dedicated hardware for CNNs.  
  **Constraints**:
  - Total memory for layer parameters is limited to **36 kB**.
  - The chip does not support average pooling, requiring an alternative approach to achieve similar functionality.

### Task B_2
- **Objective**: In addition to the constraints in Task B_1, the chip does not support floating-point operations.

---

## References

- Dataset: [Deep-Soli Dataset on Kaggle](https://www.kaggle.com/datasets/chandragupta0001/soli-data)
- Research Paper: [Google Soli - ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/2984511.2984565)

---

This repository provides tools and resources to explore and implement gesture recognition using the Google Soli radar technology. For more details, refer to the Jupyter notebooks and dataset link provided above.
