import os
import numpy as np
import pandas as pd

def adapt_ocr(ocr_root):
    image_id = []
    ocr_info = []

    for ocr_file in os.listdir(ocr_root):
        ocr_info_ = []
        image_id.append(int(ocr_file[:-4]))

        o = np.load(os.path.join(ocr_root, ocr_file), allow_pickle=True)
        texts = o.tolist()['texts']
        pre_bboxes = o.tolist()['boxes'].tolist()
        width = o.tolist()['width']
        height = o.tolist()['height']


        for i in range(len(texts)):
            word = texts[i]
            top_left_x = int(pre_bboxes[i][0]*1.0/width*1000)
            top_left_y = int(pre_bboxes[i][1]*1.0/height*1000)
            bottom_right_x = int(pre_bboxes[i][2]*1.0/width*1000)
            bottom_right_y = int(pre_bboxes[i][3]*1.0/height*1000)

            ocr_info_.append({"word": word, "bounding_box": {"top_left_x": top_left_x, "top_left_y": top_left_y, "bottom_right_x": bottom_right_x, "bottom_right_y": bottom_right_y}})
        ocr_info.append(ocr_info_)
    
    return pd.DataFrame({'image_id':image_id, 'ocr_info':ocr_info})