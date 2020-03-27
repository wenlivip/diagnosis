# diagnosis
Implementation of “Generating Diagnostic report for Ultrasound Image by High-middle-level Visual Information Incorporation on Double Deep Learning Models”

Deep Learning Platform: Tensorflow_GPU 1.13
OpenCV: 3.3

This project is divided into three folders:

detection model
  "dataset" : used to process data as the standard format, including coco and voc format.
  "layer_utils": functions of RPN, including generating anchors and ROI.
  "model": 1. save and load trained parameters.   2. implementation of data loading, gradient calculating, parameters optimizing. 
  "nets": some networks can be used as the feature extrator in detecttion model.
  "nms": mutiple implementations of non maximum suppression algorithm, which can remove redundant lesion areas in detected results.

diagnositc report generation model
  The "model.py" file contains the structure of the diagnositc report generation model.

pretrained model
  Training feature extraction model, we use resnet-50 as the extractor in this paper.
