# pytorch-multi-label-classifier

## Introdution
***

A [pytorch](https://github.com/pytorch/pytorch) implemented classifier for Multiple-Label classification. 
You can easily ```train```, ```test``` your multi-label classification model and ```visualize``` the training process.  
Below is an example visualizing the training of one-label classifier. If you have more than one attributes, no doubt than all the loss and accuracy curves of each attribute will show on web browser orderly.

Loss             |  Accuracy
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/7804678/38625748-bfdd53d2-3ddd-11e8-8993-8b1e7635e00e.png)  |  ![](https://user-images.githubusercontent.com/7804678/38625746-be8c3962-3ddd-11e8-87a0-3fbbaa1e2ee0.png)

## Module
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

## TODO
***
- [ ] Snapshot loss and accuracy records
- [ ] Support visualize multi top K accuracy
- [ ] Support model finetuning
- [ ] Complete test module


## Dependence
***
- Visdom
- Pytorch

## Reference
***
Part of codes and models refer to some other OSS listed belows for thanks:
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [pytorch-LightCNN](https://github.com/AlfredXiangWu/LightCNN)
