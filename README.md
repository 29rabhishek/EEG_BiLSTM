Implementaion of paper "EEG-based image classification via a region-level stacked bi-directional deep learning framework"


**_Training Mode_:**

**Configurations 1**: 
- *Lateralization*: Region-level Info.
-  *Feature Encoder*: Unidirectional LSTM
-  *Classifier*: Softmax

**Configurations 2**: 
- *Lateralization*: Region-level Info.
-  *Feature Encoder*: Bidirectional LSTM
-  *Classifier*: Softmax

**Configurations 6**:
-*inference script*: Infer_confif_ICA_SVM
- *Lateralization*: Region-level Info.
-  *Feature Encoder*: Stacked Bidirectional LSTM
-  *Classifier*: ICA+SVM
