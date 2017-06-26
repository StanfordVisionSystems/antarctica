import cv2
import numpy as np

def autocrop(img, thres=200):
    if( len(img.shape) != 2 ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    ret, img_thres = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    _, bound_box, _ = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x,y,w,h = cv2.boundingRect(bound_box[0])

    return img[y:y+h,x:x+w]

def detect_sprocket_holes(img):
    sp = cv2.imread('/home/ubuntu/antarctica/data/sprocket_hole2.png')

    res = cv2.matchTemplate(img, sp, cv2.TM_CCOEFF_NORMED)
    locs = np.argwhere(res > 0.9)
    print(locs[:])
    print(len(locs))
    res = 128 * res + 128
    res[res < 225] = 0
    cv2.imwrite('res.png', res)
    
def detect_textline(img):
    pass

def recognize_textline(img):
    line_word_boxes = tool.image_to_string(
        img,
        lang="glacierdigits",
        builder=pyocr.builders.WordBoxBuilder()
    )

    # returns list of objects with `position` and `content` fields
    return line_word_boxes
