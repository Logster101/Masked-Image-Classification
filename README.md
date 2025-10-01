# Face Mask Detection using MobileNetV2

## Overview
This project implements a **face mask detection model** using **MobileNetV2**. The model classifies faces into three categories:

- `with_mask` – wearing a mask correctly
- `without_mask` – not wearing a mask
- `mask_weared_incorrect` – wearing a mask incorrectly

The model is trained on annotated images and evaluated using classification metrics.

## Dataset
- **Images:** `images/`
- **Annotations (XML):** `annotations/`
- Each XML file contains bounding boxes and labels for the faces in the images.

## Requirements
- Python 3.9+
- TensorFlow 2.x
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies via:

```
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Update the paths in the script:

```python
imagePath = r'C:\path\to\images'
annotationPath = r'C:\path\to\annotations'
```

### 2. Generate CSV and Extract Faces
```python
dataTable = generate_csv_data(annotationPath)
data, labels = extract_faces(imagePath, dataTable, maskStates)
```

### 3. Preprocess Labels
```python
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
```

### 4. Split Dataset
```python
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
```

### 5. Build and Train the Model
```python
model = build_model()
epochs = train_model(model, trainX, testX, trainY, testY, numEpochs=25, lr=1e-4, bs=1)
```

### 6. Evaluate Model
```python
predictedVals = model.predict(testX)
predictedVals = np.argmax(predictedVals, axis=1)
trueVals = testY.argmax(axis=1)
report = classification_report(trueVals, predictedVals, target_names=lb.classes_)
print(report)
plt_confusion_matrix(trueVals, predictedVals)
```

### 7. Test Predictions
```python
for i in range(10):
    x = random.randint(0, len(data)-1)
    p = model.predict(data[x:x+1])
    p = np.argmax(p, axis=1)
    print("Actual:", labels[x], "Predicted:", maskStates[p[0]])
```

## Results
- Test accuracy: ~91%
- Confusion matrix shows strong performance on `with_mask` and `without_mask` classes
- `mask_weared_incorrect` is harder to classify due to fewer samples

## License
This project is for educational use.
