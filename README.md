# Implement Pets (Cat/Dog) Detection
----

## 1. Information
* This model is trained for 7 classes (2 for pets, 6 for human body parts)
* check file ```pets_nake.yaml``` for more detail about each class

## 2. Usage
* run detect.py

```
bash run_detect.sh
```

### <span style="color:red">Note<span>
* set ```--classes argument to 0 1 (for cat and dog only)``` otherwise the model will detect other classes
* output image(s) is/are saved in runs/detect/ folder
