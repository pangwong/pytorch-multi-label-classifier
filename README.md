# pytorch-multi-label-classifier

## Introdution

A [pytorch](https://github.com/pytorch/pytorch) implemented classifier for Multiple-Label classification. 
You can easily ```train```, ```test``` your multi-label classification model and ```visualize``` the training process.  
Below is an example visualizing the training of one-label classifier. If you have more than one attributes, no doubt than all the loss and accuracy curves of each attribute will show on web browser orderly.

Loss             |  Accuracy
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/7804678/38625748-bfdd53d2-3ddd-11e8-8993-8b1e7635e00e.png)  |  ![](https://user-images.githubusercontent.com/7804678/38625746-be8c3962-3ddd-11e8-87a0-3fbbaa1e2ee0.png)

## Module

- ### ```data```
  data preparation module consisting of reading and transforming data. All data store in ```data.txt```and ```label.txt``` with some predefined format explained below.
- ### ```model```
  scripts to build multi-label classifier model. Your model templets should put here.
- ### ```options```
  train test and visualization options define here
- ### ```util```
  - ```webvisualizer```: a [visdom](https://github.com/facebookresearch/visdom) based visualization tool for visualizing loss and accuracy of each attribute
  - ```util```: miscellaneous functions used in project
  - ```html```: used in webvisualizer.
- ### ```test``` 
  - ```mnist```: [mnist](http://yann.lecun.com/exdb/mnist/) dataset arranged as defined data format.
  - ```celeba```: exactract some of attributes of [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Multi-Label Data Format

Data Format Explanation. 
- ```label.txt```

  Store attribute information including its name and value. [label.txt example](https://github.com/pangwong/pytorch-multi-label-classifier/blob/master/test/celeba/label.txt). Lines in ```label.txt``` stack as follows: 
  
  > - For each ```attribute``` :
  >   - ```number of attribute values``` ; ```id of attribute``` ; ```name attribute``` 
  >   - For each ```attribute value``` belonging to current ```attribute``` :
  >     - ```id of attibute_value``` ; ```name of attribute value```
  >
  Note: mind the difference between attribute and attribute value.
- ```data.txt``` 

  Store objects information including attribute id and bounding box and so on. Each line is one json dict recording one object. [data.txt example](https://github.com/pangwong/pytorch-multi-label-classifier/blob/master/test/celeba/data.txt)
  
  >
  > - ```"box"```:object boundingbox. ```'x'```: top_left.x , ```'y'```:top_left.y, ```'w'```: width of box, ```'h'```: height of box.
  > - ```"image_id"```: image identifier. An image content dependent hash value.
  > - ```"box_id"```: object identidier. Combine ```image_id```, ```box['x']```, ```box['y']```, ```box['w']```, ```box["h"]``` with ```_```.
  > - ```"size"```: image width and height. Used for varifying whether box is valid. 
  > - ```"id"```: list of ids. Store multi-label attributes ids, the order is the same as the attributes' order in ```label.txt```

## Dependence

- Visdom
- Pytorch


## TODO

- [ ] Snapshot loss and accuracy records
- [ ] Support visualize multi top K accuracy
- [x] Support model finetuning
- [x] Complete test module
- [ ] Add switch to control loss and accuracy curves displaying on one plot or multiple
- [ ] Train and Test Log


## Reference

Part of codes and models refer to some other OSS listed belows for thanks:
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [pytorch-LightCNN](https://github.com/AlfredXiangWu/LightCNN)
