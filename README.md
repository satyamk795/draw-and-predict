# draw-and-predict
This is a web-based object recognition app that lets users draw on a canvas. A trained CNN model then predicts what the user drew, based on Google’s Quick, Draw! dataset.

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript (Canvas API)
- **Backend:** Python (Flask)
- **ML Framework:** TensorFlow/Keras
- **Model Input:** 28x28 grayscale image
- **Dataset Used:** [Quick, Draw! Dataset by Google](https://quickdraw.withgoogle.com/data)

## Model Details

- **Model Type:** Convolutional Neural Network (CNN)
- **Dataset:** Quick, Draw! 
- **Input:** 28x28 image (from canvas)
- **Output:** Object class (e.g.,airplane, apple, house.)
- **Accuracy:** Varies per category; tested on validation data
- **Framework:** TensorFlow + Keras

