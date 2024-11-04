# IMCFN
IMCFN: Image-based malware classification using fine-tuned convolutional neural network architecture.


Malware Image Classification

Project Overview

This project focuses on classifying malware images using deep learning models. By converting malware binaries into visual representations, the project aims to classify malware families based on these images, leveraging multiple Convolutional Neural Network (CNN) architectures. This approach is part of a broader effort to use visual data analysis in cybersecurity, offering an innovative method to detect and categorize malware types.

Key Features

	•	Multi-Model Evaluation: The project compares several popular deep learning architectures, including IMCFN (Interleaved Multi-Filter Convolutional Network), InceptionV3, ResNet50, and VGG16.
	•	Color and Grayscale Analysis: Each model is evaluated using both color and grayscale image datasets to understand the impact of color channels on classification accuracy.
	•	Image Augmentation: Configurations include models trained with and without image augmentation to explore its effects on model generalization and performance.
	•	Visualization and Analysis: A dedicated notebook (Graphs.ipynb) provides visualizations, such as accuracy and loss graphs, confusion matrices, and model comparison metrics, for comprehensive analysis.

Project Structure

	•	IMCFN Model: Includes configurations for both color and grayscale images, leveraging the IMCFN architecture tailored for malware image classification.
	•	InceptionV3 Model: Tested with color and grayscale versions of the dataset to evaluate deep feature extraction in malware analysis.
	•	ResNet50 Model: Incorporates ResNet50 to utilize deep residual learning for improved classification.
	•	Original VGG16 Model: Uses the VGG16 architecture with and without image augmentation, allowing a comparison of augmentation effects on performance.

Technical Requirements

	•	TensorFlow and Keras: Core libraries for model building and training.
	•	GPU Optimization: GPU support is integrated to handle large datasets and computationally intensive CNN architectures efficiently.
	•	Python Libraries: Includes additional libraries for data manipulation, visualization, and evaluation (e.g., NumPy, pandas, matplotlib).

How It Works

	1.	Data Preparation: Malware binaries are converted into image formats, which are then categorized by malware family. The dataset is split into training and testing sets.
	2.	Model Training: Each model architecture is trained separately on both color and grayscale images. Models with image augmentation are also trained to assess generalization.
	3.	Performance Evaluation: The trained models are evaluated on a test set, with metrics like accuracy, precision, and recall used to assess their efficacy in classifying malware families.


Conclusion

This project demonstrates the potential of using CNNs to classify malware by analyzing visual representations. By comparing multiple architectures, color channels, and augmentation techniques, it provides insights into the best practices for implementing CNN-based malware classifiers in cybersecurity applications.

Cite This Work:

@article{Vasan2020IMCFN,
  title     = {IMCFN: Image-based malware classification using fine-tuned convolutional neural network architecture},
  author    = {Danish Vasan and Mamoun Alazab and Sobia Wassan and Hamad Naeem and Babak Safaei and Qin Zheng},
  journal   = {Computer Networks},
  volume    = {171},
  pages     = {107138},
  year      = {2020},
  issn      = {1389-1286},
  doi       = {10.1016/j.comnet.2020.107138},
  url       = {https://www.sciencedirect.com/science/article/pii/S1389128619304736},
  abstract  = {The volume, type, and sophistication of malware is increasing. Deep convolutional neural networks (CNNs) have lately proven their effectiveness in malware binary detection through image classification. In this paper, we propose a novel classifier to detect variants of malware families and improve malware detection using CNN-based deep learning architecture, called IMCFN (Image-based Malware Classification using Fine-tuned Convolutional Neural Network Architecture). Differing from existing solutions, we propose a new method for multiclass classification problems. Our proposed method converts the raw malware binaries into color images that are used by the fine-tuned CNN architecture to detect and identify malware families. Our method previously trained with the ImageNet dataset (≥10 million) and utilized the data augmentation to handle the imbalance dataset during the fine-tuning process. For evaluations, an extensive experiment was conducted using 2 datasets: Malimg malware dataset (9,435 samples), and IoT- android mobile dataset (14,733 malware and 2,486 benign samples). Empirical evidence has shown that the IMCFN stands out among the deep learning models including other CNN models with an accuracy of 98.82% in Malimg malware dataset and more than 97.35% for IoT-android mobile dataset. Furthermore, it demonstrates that colored malware dataset performed better in terms of accuracy than grayscale malware images. We compared the performance of IMCFN with the three architectures VGG16, ResNet50 and Google's InceptionV3. We found that our method can effectively detect hidden code, obfuscated malware and malware family variants with little run-time. Our method is resilient to straight forward obfuscation technique commonly used by hackers to disguise malware such as encryption and packing.},
  keywords  = {Cybersecurity; Malware; Image-based malware detection; Convolutional neural network; Transfer learned; Fine-tuned; Deep Learning; Obfuscation; IoT-Android Mobile}
}
