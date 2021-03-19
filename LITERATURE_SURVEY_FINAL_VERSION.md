## Abstract

## Introduction

## Design Methodologies
  ### Datatsets
  In this subsection, we introduce the sources of data made use of in all the related works surveyed. There are numerous large datasets available containing thousands of images of galaxies. Most of the datasets contain galaxies imaged by the Sloan Digital Sky Survey (SDSS).
  
  
In [1], an attempt was made to manually collected their images from Google using a browser extension called Fatkun Batch Image Download which led them to do a significant amount of pre-processing, discussed in the following subsection. Some of this pre-processing could be avoided by using datasets like the one by Galaxy Zoo. The current Galaxy Zoo combines images from the SDSS with the most distant images yet from Hubble's CANDELS project. The GZ data has fuelled multiple papers. The GZ data comes with 61578 pre-classified 424 × 424 RGB images of galaxies taken from SDSS. It also comes with the classifications that have been collected from a crowdsourced quiz which are given in the form of probabilities of the answers to 37 questions. This dataset was used in [2] and [4]. In [3], a dataset obtained from Zsolt Frei’s galaxy catalogue was used. Hubble Tuning Fork images were used as model or prototype images in [5].
In [6],Sloan Digital Sky Survey (SDSS) Data set was used. In [7],the main dataset was employed from Sloan Digital Sky Survey Data Release 7 (SDSS-DR7) and Galaxy Zoo catalogues with 670,560 galaxies for measuring morphology and training the classification models. In [8],The dataset used was taken from the EFIGI catalogue This catalogue dataset consists of more than 11,000 images and contains samples from different Hubble types galaxies. This catalogue also combines the data from standard surveys and catalogues of Sloan Digital Sky Survey, Value-Added Galaxy Catalogue, Principal Galaxy Catalogue. In [9],The galaxy images in this are drawn from the galaxy zoo which contains 61578 JPG colour galaxy images where each image is 424x424x3 pixels in size with probability that each galaxy is is classified into different morphologies. In [10],full sample sets of Galaxy images from SDSS DR7 was used. The dataset consists of 667,935 entries each of which corresponds to an object in the SDSS database. After some sort of filtering, the dataset was reduced to 251,867.
[11] uses Galaxy Zoo 2 and Nair et al. catalogues 2010 for training .For testing the catalogues used were Huertas-Company et al. 2011 and Cheng et al. 2011 catalogues. Southern Photometric Local Universe Survey (S-PLUS) is an imaging survey that southern sky, it uses 12 optical bands. [12] uses the first release of the S-PLUS which includes both images and catalogues of detected objects and the labels for this dataset is obtained by matching the astronomical coordinates from S-PLUS to SDSS spectra catalogue
[13] uses Galaxy Zoo 2 catalog and datasets derived from Catalog Archive Server of SDSS(Sloan Digital Sky Survey) where the image size was 120x120 px. An approximate of 2,45,609 images were used[13].Zsolt frei Catalog comprises of nearly 113 images from nearby galaxies taken in multiple pass bands the images had high resolution and had better calibrations therefore extensive pre-processing was not required[14].



  ### Preprocessing
  
  ### Models
  The goal of this sub-section is to provide insight into the techniques, architectures and models that we encountered in the papers surveyed in this work. Many algorithms of Machine learning and Deep learning have been used. Machine learning is a field of research that deals with learning algorithms used to parse and learn from data and make predictions based on the same. Deep Learning is a subfield of machine learning concerned with algorithms that create artificial neural networks to make decisions. Convolutional Neural Networks are a type of deep neural networks which are extensively used for image recognition and classification.
  
## Results

## Discussions
