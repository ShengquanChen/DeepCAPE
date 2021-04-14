# DeepCAPE
DeepCAPE is a deep convolutional neural network to predict enhancers via the integration of DNA sequences and corresponding DNase-seq data.

# Requirements
- keras
- numpy
- hickle
- random
- tensorflow

# Installation
Download DeepCAPE by
```shell
git clone https://github.com/ShengquanChen/DeepCAPE
```
```
 Arguments:  
  inputfile: preprocessed data (hkl format)  
  outputfile: trained model (h5 format)
```
Installation has been tested in a Linux/MacOS platform with Python2.7.

# OpenAnnotate
Use OpenAnnotate (http://health.tsinghua.edu.cn/openannotate/) to annotate the chromatin accessibility of genomic regions across diverse types of cell lines, tissues, and systems. We also provide a simplified [demo](http://health.tsinghua.edu.cn/openness/anno/info/demos/RegulatoryMechanism/RegulatoryMechanism.html) in OpenAnnotate for the demonstration of DeepCAPE pipeline.

# Citation
Chen, Shengquan, Mingxin Gan, Hairong Lv, and Rui Jiang. "DeepCAPE: a deep convolutional neural network for the accurate prediction of enhancers." Genomics, Proteomics & Bioinformatics (2021).

# License
This project is licensed under the MIT License - see the LICENSE file for details
