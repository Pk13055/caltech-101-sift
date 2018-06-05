# Caltech-101 5-class classification

## Data

- Data consists of a training dataset consisting of 1000 images, intersparsed
  between:
    - airplanes 
    - bikes
    - faces
    - leopards
    - watches
- The dimensions of the dataset are (1000, 243), 243 stands for the word to vec
  encoding of the descriptors for each image.
- 243 clusters of the SIFT features were taken and clustering was performed.
    - The 243-repr of the input represents the scaled count of the number of SIFT
      features per cluster.
    - This gives a homogeneous representation of the input irrespective of the
      number of SIFT features per image.

## Models

- The data is trained using both `kNN` and `SVM` (_linear and gaussian kernel_).
- Standard python machine learning libraries have been used.
    - sklearn (For SVM, SVR, LinearSVC, LinearSVR)
    - numpy (data manipulation)
    - pandas (intermediate and long term storage)
    - h5py (dense effficient storage)
- None of the hyperparameters have been changed. Also, given that the
  `test_data` is just as big as the training, **and** that the input vector size
  is much less than the training examples, data is dense enough to prevent
  overfitting.

## Results

| Method      | Score (%) |
|  :---:      |   :---:   |
|  **kNN**    |   70.00*  |
|  **kNN** (1v1) | 70.5*  |
|  SVM (1v1)  |   55.00   | 
|  SVM (1vmany)       |   16.50   |
|  Linear SVM |   63.00   |
| Linear SVM (1v1)| 45.00 | 
|  radius-NN  |   16.50   |
