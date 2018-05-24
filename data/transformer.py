import copy
from PIL import Image
from torchvision import transforms


def get_transformer(opt):
    transform_list = []
    
    # resize  
    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    
    # grayscale
    if opt.input_channel == 1:
        transform_list.append(transforms.Grayscale())

    # crop
    if opt.crop == "RandomCrop":
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.crop == "CenterCrop":
        transform_list.append(transforms.CenterCrop(opt.input_size))
    elif opt.crop == "FiveCrop":
        transform_list.append(transforms.FiveCrop(opt.input_size))
    elif opt.crop == "TenCrop":
        transform_list.append(transforms.TenCrop(opt.input_size))
    
    # flip
    if opt.mode == "Train" and opt.flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    transform_list.append(transforms.ToTensor())
    
    # If you make changes here, you should also modified 
    # function `tensor2im` in util/util.py accordingly
    transform_list.append(transforms.Normalize(opt.mean, opt.std))

    return transforms.Compose(transform_list)

def fix_box(box, width, height, ratio=-1, scale=1.0):
    if scale < 0:
        scale = 1.0
    box = copy.deepcopy(box)
    w = box["w"]
    h = box["h"]
    x = box["x"] + w / 2
    y = box["y"] + h / 2
    mw = 2 * min(x, width - x)
    mh = 2 * min(y, height - y)
    w = max(1, min(int(w * scale), mw))
    h = max(1, min(int(h * scale), mh))
    if ratio > 0:
      if 1.0 * w / h > ratio:
          h = int(w / ratio)
          h = min(h, mh)
          w = int(h * ratio)
      else:
          w = int(h * ratio)
          w = min(w, mw)
          h = int(w / ratio)
    box["x"] = x - w / 2
    box["y"] = y - h / 2
    box["w"] = w
    box["h"] = h
    return box

def load_image(image_file, box, opt, transformer):
    img = Image.open(image_file)
    if opt.input_channel == 3:
        img = img.convert('RGB')
    width, height = img.size
    # box crop
    if box is not None and opt.region == True:
        box = fix_box(box, width, height, opt.box_ratio, opt.box_scale)
        area = (box['x'], box['y'], box['x']+box['w'], box['y']+box['h'])
        img = img.crop(area)
    # transform
    input = transformer(img)
    return input

