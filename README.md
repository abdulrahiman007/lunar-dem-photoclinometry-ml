# lunar-dem-photoclinometry-ml
Generate lunar Digital Elevation Models (DEMs) from a single image using photoclinometry with optional ML refinement.

# Introduction
This project presents a lunar terrain analysis system that generates Digital Elevation Models (DEMs) from a single grayscale lunar surface image using a photoclinometry (shape-from-shading) approach.
A DEM represents the relative elevation and surface structure of terrain, highlighting features such as craters, slopes, and boulders.
The system combines a physics-based elevation estimation pipeline with a lightweight Random Forest–based machine learning refinement step to improve surface consistency and visualization quality.
The project is implemented as a web-based application using Flask, allowing users to upload an image and generate DEM outputs instantly.
# Core Methodology
1.Photoclinometry (Shape-from-Shading)
Assumes a sun direction (azimuth & elevation)
Analyzes brightness variations caused by terrain slopes
Computes surface gradients using image processing techniques
Integrates gradients to estimate relative elevation
Applies bilateral filtering to reduce noise
Produces grayscale and color-mapped DEM outputs

2.Machine Learning Refinement (Random Forest)
ML is applied as a post-processing refinement step
Converts image and elevation information into numerical features
Random Forest predicts local correction values
Improves surface smoothness and consistency
Does not replace the physics-based model
# Features
Single-image DEM generation
Photoclinometry-based elevation estimation
Noise reduction using edge-preserving filtering
Color-mapped visualization for terrain clarity
Random Forest refinement module
Web-based interface for easy interaction
Modular and extendable architecture
# How to Run
1️ Install Dependencies
pip install flask opencv-python numpy scikit-learn joblib

2️ Run the Application
python app.py

3️ Open in Browser
Visit:
http://127.0.0.1:5000/
Upload a lunar image to generate DEM outputs.

4️ Train Random Forest (Optional)
If training is required:
python ml/train_rf.py
After training, ML refinement will be applied automatically during processing.
# Project Directory Structure
```
Lunar-DEM-Project/
│
├── app.py
├── processor.py
│
├── logic/
│   ├── sun_model.py
│   ├── gradient_map.py
│   └── depth_estimator.py
│
├── ml/
│   ├── train_rf.py
│   ├── predict_rf.py
│   └── feature_extractor.py
│
├── static/
│   ├── input/
│   └── output/
│
├── utils/
│   ├── height_and_path_demo.py
│   └── visualization.py
│
├── templates/
│   └── index.html
│
└── README.md
```
# Applications & Future Scope
Applications
Preliminary lunar terrain analysis
Surface slope visualization
Academic demonstrations
Educational research projects
# Future Enhancements
Integration of real sun-angle metadata
Support for planetary datasets (Mars, asteroids)
Super-resolution DEM enhancement
Hybrid physics + deep learning refinement
Integration with rover navigation simulation systems
# Technical Highlights
Modular architecture separating UI, processing, and ML layers
Physics-based DEM generation pipeline
ML refinement without over-dependence on large datasets
Lightweight and computationally efficient design
# Author

Abdul Rahiman  | Computer Science Student
