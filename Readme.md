# Project Title: Predicting the Severity of Damage Done by Natural Disasters Using Deep Learning

## Project Overview

The aim of this project is to create a predictive model that evaluates the severity of damage caused by natural disasters through deep learning techniques. By utilizing advanced machine learning algorithms, the project seeks to deliver precise and timely predictions to support disaster response and resource allocation.

## Approach

- **Ensemble of Semantic Segmentation Models**: Implementing an ensemble approach to improve prediction accuracy by combining multiple models.
- **Shared Encoder Architecture**: Utilizing a shared encoder for both pre- and post-disaster images, with extracted features concatenated and fed into the decoder.
- **Model Variety**: Employing a variety of encoders such as ResNets, DenseNets, and EfficientNets, along with two decoders: U-Net and Feature Pyramid Network (FPN).
- **Ensemble Optimization**: Using weighted averaging for ensemble predictions, with weights optimized for each model based on validation data.
- **Severity Index Calculation**: Creating masks of predicted images to generate a post-disaster mask, which will be used to calculate the severity index.
- **Cycle GAN Utilization**: Leveraging Cycle GAN to generate images with highlighted affected regions (work in progress).

## Credit

- This project builds upon the [xView2-Solution](https://github.com/BloodAxe/xView2-Solution) repository.
- Pretrained models can be downloaded from the release tab of the aforementioned repository.
        