import os
import time
import collections
import numpy as np
from . import util
from . import html


class WebVisualizer():
    def __init__(self, opt):
        self.opt = opt
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.use_html = (opt.html and (opt.mode == "Train"))
        self.name = opt.name
        self.saved = False
        self.type2id = {"Loss":0, "Accuracy": 1, "Other": 2}
        self.phase2id = {"Train": 0, "Validate": 1, "Test": 2}
        
        def ManualType():
            return collections.defaultdict(list)
        # store all the points for regular backup 
        self.plot_data = collections.defaultdict(ManualType)
        # line window info 
        self.win_info = collections.defaultdict(ManualType)
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)
        
        if self.use_html:
            self.web_dir = os.path.join(opt.model_dir, "web")
            self.img_dir = os.path.join(opt.model_dir, "image")
            print "Create web directory %s ..." %(self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
            

    def reset(self):
        self.saved = False
    
    """
    type:  [Accuracy | Loss | Other]
    phase: [Train | Validate | Test]
    """
    def plot_points(self, x, y, data_type, phase):
        line_name = data_type + "@" + phase
        self.plot_data[data_type][phase].append((x,y))
        # draw ininial line objects if not initialized
        if len(self.win_info[data_type][phase]) == 0:
            for index in range(len(y)):
                win_id = self.type2id[data_type]*len(y) + index
                win = self.vis.line(X=np.array([0]),
                                    Y=np.array([0]),
                                    opts=dict(
                                        title=data_type + " of Attribute " + str(index) + " Over Time",
                                        xlabel="epoch",
                                        ylabel=data_type,
                                        showlegend=True,
                                        width=900,
                                        height=450),
                                    win=win_id,
                                    name=line_name)
                self.win_info[data_type][phase].append(win)
        
        for index, value in enumerate(y): 
            win_id = self.win_info[data_type][phase][index] 
            self.vis.line(X=np.array([x]),
                          Y=np.array([value]),
                          win=win_id,
                          name=line_name,
                          update="append")
    
    def plot_images(self, image_dict, start_display_id, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.image_ncols
            if ncols > 0:
                h, w = next(iter(image_dict.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(image_dict.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in image_dict.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=start_display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=start_display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in image_dict.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=start_display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in image_dict.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in image_dict.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def backup(self, name):
        pass

    def test(self):
        pass
