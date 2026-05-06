# ECG Preprocessing Pipelines: Old vs. New

This document explains the differences between the old and new data preprocessing pipelines used for loading ECG signals, focusing on why the transition was necessary for handling the pediatric dataset.

## The Two Pipelines

### 1. Old Pipeline (Resampling/Interpolation)
**Methodology:** 
The old preprocessing pipeline relied on the `resample_unequal` function, which used linear interpolation to stretch or compress every input signal to exactly 5000 data points.
- It read the `Sampling_point` (sampling rate) from the dataset metadata.
- It interpolated the signal across time to force the output array into a shape of `(12, 5000)`.

**Why it was used previously:**
The original Harvard dataset consisted of recordings that were all exactly 10 seconds long, but acquired at varying sampling rates (e.g., 250 Hz or 500 Hz). Interpolating a 10-second signal at 250 Hz (2500 points) to 5000 points effectively upsamples it to 500 Hz without altering the true physical duration of the recording.

### 2. New Pipeline (Cropping/Padding)
**Methodology:**
The new pipeline removes the linear interpolation and instead enforces length via truncation (cropping) or zero-padding.
- If the signal is longer than 5000 samples, it is cropped to the first 5000 samples.
- If the signal is shorter than 5000 samples, it is padded with trailing zeros to reach 5000 samples.

## Why We Switched for the Pediatric Dataset

The shift to the new pipeline was strictly required due to the varying recording durations of the pediatric dataset.

### The Problem with Interpolation on Varying Durations
In the pediatric dataset (ZU pECG) and CinC2021 datasets, the sampling rate is consistently **500 Hz**. However, the **physical duration of the recordings varies greatly**:
- ECGFounder was primarily pretrained on 10-second recordings (5000 points at 500 Hz).
- The pediatric dataset contains recordings ranging from 5 to 120 seconds, with the vast majority being **30 seconds long** (15,000 points at 500 Hz).

If we applied the **old pipeline** to a 30-second pediatric recording:
1. The code would take 15,000 points and forcefully interpolate them down to 5000 points.
2. Because the sampling rate is constant, this does not "resample" the frequency—it functionally speeds up time by a factor of 3.
3. A 30-second heart rhythm would be compressed into a 10-second window. This would artificially triple the apparent heart rate and severely distort crucial morphological features like the QRS complex width, making the signal unrecognizable to the foundation model.

### The Solution: Truncation
By switching to the **new pipeline**, we preserve the true physical time domain of the signal:
- Because the sampling rate is already correct (500 Hz), 1 data point = 2 milliseconds.
- By taking the first 5000 points via array slicing (`data[:, :5000]`), we extract exactly the first **10 seconds** of the ECG recording.
- This ensures that the heart rate, complex durations, and signal morphology perfectly match the 10-second format that ECGFounder was trained to expect.
