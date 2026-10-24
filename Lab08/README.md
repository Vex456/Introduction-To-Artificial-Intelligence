# Lab 08 — Handwritten Digit Recognition with Convolutional Neural Networks (CNN)

This laboratory implements a Convolutional Neural Network (CNN) in **TensorFlow / Keras** to classify 28×28 grayscale images of handwritten digits (0–9) using the Kaggle MNIST Digit Recognizer dataset.

---

## 1. Network Architecture

The Sequential model comprises two convolutional blocks followed by a dense classification head:

```
Input (28, 28, 1)
  │
  ├── Conv2D (32 filters, 3x3) + ReLU
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (64 filters, 3x3) + ReLU
  ├── MaxPooling2D (2x2)
  │
  ├── Flatten
  ├── Dense (128 units) + ReLU
  ├── Dropout (0.3)
  └── Dense (10 units) + Softmax  ──> Digits 0 to 9
```

---

## 2. Methodology & Preprocessing

1. **Normalization:** Raw integer pixel intensities in $[0, 255]$ are scaled to the $[0.0, 1.0]$ float32 range.
2. **Reshaping:** 784-dimensional flat vectors are reshaped to standard image tensor dimensions: `(-1, 28, 28, 1)`.
3. **Stratified Split:** 80% training ($33,600$ images), 20% validation ($8,400$ images) with class stratification.
4. **Optimization:** Trained with Adam optimizer and `sparse_categorical_crossentropy` loss across 10 epochs.
5. **Performance:** Achieved **~98.9% validation accuracy**.

---

## 3. Evaluation & Inference

- Validation accuracy and loss curves plotted over training epochs.
- Inference on Kaggle test dataset (`test.csv`).
- Visualized sample test predictions with grayscale digit plots.
- External image loading and preprocessing pipeline with Pillow (`PIL`).
