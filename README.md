# pytorch-multi-label-classifier


#### Introdution
---
A [pytorch](https://github.com/pytorch/pytorch) implemented classifier for Multiple-Label classification. 
You can easily ```train```, ```test``` and ```visualize``` your multi-label classification model.

### Module
***
- ### data
  data preparation module consisting of reading and transforming data. All data store in ```a.txt```and ```label.txt``` with some predefined format.
- ### model
  scripts to build multi-label classifier model. Your model templets should put here.
- ### options
  train test and visualization options define here
- ### util
  - webvisualizer: a [visdom](https://github.com/facebookresearch/visdom) based visualization tool for visualizing loss and accuracy of each attribute
  - util: miscellaneous used in project
  - html: used in webvisualizer.

### TODO
***
- [ ] Snapshot loss and accuracy recode
- [ ] Support visualize multi top K accuracy
- [ ] Support model finetuning

### Example
***

 
### Reference
***
Some of codes and models refer to other OSS listed belows for thanks:
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [pytorch-LightCNN](https://github.com/AlfredXiangWu/LightCNN)
