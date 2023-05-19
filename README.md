# Stress Detection using PPG Sensor and Machine Learning

This project aims to develop a machine learning model capable of accurately determining a person's stress levels based on readings from a PPG (Photoplethysmography) sensor, which is connected to a microcontroller such as Arduino Uno. By extracting 34 different data variables, including Time Domain, Frequency Domain, and Non-Linear Domain features, from two sensor readings (Beats Per Minute and InterBeat Intervals), we can assess stress levels. 

This is however just a bare implementation of the machine learning. Do not use the code for any medical or scientific results. This is a part of the main project.

To obtain certain values like Higuchi fractal dimensions, we utilized a Python library called Antropy, while the hrv-analysis library helped derive specific Frequency Domain data variables and the Sample Entropy value. The remaining variables were defined as functions within the program using mathematical and statistical formulas.

With the collection of diverse data variables, we trained a neural network on the [SWELL dataset](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv) consisting of 264,139 sets of readings. The machine learning model achieved a remarkable accuracy of over 99% when tested against both the training and test data. Once the sensor readings were processed through the model, it provided a result indicating whether the user was experiencing stress or not.

## Getting Started

To replicate or build upon this project, follow the instructions below:

1. Set up the hardware components, including the PPG sensor and Arduino Uno.
2. Install the required libraries, such as Antropy and hrv-analysis.
3. Clone or download the project repository from GitHub.
4. Connect the PPG sensor to the microcontroller.
5. Upload the Arduino code to the microcontroller.
6. Run the Python program for data processing and machine learning model implementation.
7. Interpret the model's output to determine the user's stress levels.

## Dependencies

Ensure that the following dependencies are installed:

- Antropy (Python library)
- hrv-analysis (Python library)
- Other necessary Python libraries (NumPy, Pandas, TensorFlow, etc.)

## Usage

1. Modify the Arduino code to accommodate your specific PPG sensor and microcontroller setup.
2. Adjust the Python code to suit your dataset and feature extraction requirements, if necessary.
3. Run the Python program to process the PPG sensor readings and train the machine learning model.
4. Evaluate the model's accuracy and performance on the test dataset.
5. Apply the trained model to new sensor readings to predict stress levels.

## Results

Our machine learning model achieved an accuracy of over 99% when predicting stress levels based on the PPG sensor readings. The model was trained on a large dataset and utilized various data variables obtained through feature extraction. The provided code and instructions allow you to replicate these results and further refine the stress detection system.

## Contributing

We welcome contributions to enhance the accuracy and efficiency of the stress detection model. Feel free to submit pull requests or raise issues with any improvements or ideas you may have.

## License

This project is licensed under the [GNU GPLv3](LICENSE). You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## Acknowledgements

We would like to express our gratitude to the authors and maintainers of Antropy and hrv-analysis libraries, whose work greatly contributed to this project. We also acknowledge the support and resources provided by the open-source community.


