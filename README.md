---
license: mit
tags:
  - image-classification
  - medical-imaging
  - pneumonia-detection
  - deep-learning
  - tensorflow
  - resnet
  - chest-xray
pipeline_tag: image-classification
---

# 🫁 Pneumonia Detection from Chest X-Ray Images

This model detects **pneumonia** in chest X-ray images using transfer learning with a **ResNet architecture**, trained on the Kaggle Chest X-Ray Images (Pneumonia) dataset.

> ⚠️ **Disclaimer:** This model is intended for **educational and research purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional.

---

## 📌 Model Summary

| Property        | Details                              |
|----------------|--------------------------------------|
| **Task**        | Binary Image Classification          |
| **Classes**     | `Normal`, `Pneumonia`                |
| **Architecture**| ResNet (Transfer Learning)           |
| **Framework**   | TensorFlow / Keras                   |
| **Input Size**  | 180 × 180 × 3 (RGB)                  |
| **Output**      | Softmax probability over 2 classes   |
| **Model File**  | `xray_model.hdf5`                    |

---

## 📂 Dataset

The model was trained on the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Total Images:** 5,863 JPEG chest X-rays
- **Classes:** Normal, Pneumonia (Bacterial + Viral)
- **Split:**
  - Train: 5,216 images
  - Validation: 16 images
  - Test: 624 images
- **Patient Demographics:** Pediatric patients aged 1–5 years from Guangzhou Women and Children's Medical Center, Guangzhou

**Citation:**
> Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018),
> *"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification"*,
> Mendeley Data, V2, doi: [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2)

---

## 📊 Metrics

| Metric        | Score   |
|--------------|---------|
| **Accuracy**  | 91.50%  |
| **Precision** | 91.00%  |
| **Recall**    | 92.00%  |
| **F1 Score**  | 91.00%  |

> 💡 For medical imaging tasks, **Recall (Sensitivity)** is the most critical metric — a missed pneumonia case (false negative) is more dangerous than a false alarm.

---

## 🚀 How to Load & Use the Model

### Install dependencies

```bash
pip install tensorflow huggingface_hub pillow opencv-python numpy
```

### Load the model

```python
import tensorflow as tf
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="anshvsingh/pneumonia-detection",
    filename="xray_model.hdf5"
)
model = tf.keras.models.load_model(model_path)
```

### Run a prediction

```python
import numpy as np
import cv2
from PIL import Image, ImageOps

class_names = ["Normal", "Pneumonia"]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (180, 180), Image.LANCZOS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image[np.newaxis, ...]

    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])

    print("Prediction : {}".format(class_names[np.argmax(score)]))
    print("Confidence : {:.2f}%".format(100 * np.max(score)))

predict("chest_xray.jpeg")
```

---

## 🌐 Web Application

A live Streamlit web app is available for easy inference without writing any code.

👉 **GitHub Repository:** [anshvsingh/pneumonia-detection](https://github.com/anshvsingh/pneumonia-detection)

```bash
# Run locally
git clone https://github.com/anshvsingh/pneumonia-detection
cd pneumonia-detection
pip install -r requirements.txt
streamlit run xray_web.py
```

---

## 🔬 Approach

- **Transfer Learning** was applied using a pre-trained **ResNet** architecture
- The model was fine-tuned on chest X-ray images for binary classification
- Training was done on **Google Colab** using GPU acceleration
- **TensorBoard** was used to monitor training and evaluation metrics

---

## 👤 Author

**Ansh V Singh**
- GitHub: [@anshvsingh](https://github.com/anshvsingh)
- Hugging Face: [@anshvsingh](https://huggingface.co/anshvsingh)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and build upon it with attribution.
