Project Title: Automated Patient Emotion and Risk State Detection from 3D Facial Landmarks — A Precursor MARL Simulation

1. Project Objective

Develop a computer vision and ML pipeline that detects and visualizes patient emotions using 478-point 3D facial landmark data from video samples, as groundwork for later integration into a simulated MARL environment for hospital patient monitoring.

2. Core Steps and Modules

Data Loading & Preprocessing

Load the facial landmark data from Parquet/CSV files.

Parse the video files and frame-level facial coordinates.

Handle missing files and perform sanity checks.

Random Frame & Landmark Visualization

Select random frames from various videos.

Overlay 478 3D facial landmark points on corresponding video frames.

Plot frames in a grid organized by detected emotion, using Matplotlib for visualization.

Intended as both quality control and exploratory data analysis.

Emotion Classification Model

Build a lightweight classifier (e.g., CNN, LSTM, or MLP) that predicts emotion classes (e.g., Happy, Sad, Angry) directly from frame-level 3D facial landmarks.

Perform model training, validation, and evaluation.

Try time-series models if possible for improved sequential learning.

MARL Agent Environment Simulation (Extension)

Create a simulated hospital ward where multiple agents, each monitoring streams of landmark/emotion data, communicate and coordinate risk alerts.

Formulate a reward function to encourage agents to maximize early detection and minimize overload or false positives.

Evaluate agent teamwork and coordination effect on detection speed and precision.