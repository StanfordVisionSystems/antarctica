import cv2
import pickle
import pyocr
import pyocr.builders
import numpy as np
import scipy.ndimage

from skimage.feature import hog
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from PIL import Image

class BasicOCRReader:
    def __init__(self, logger):
        self.logger = logger

        self.tool = pyocr.get_available_tools()[0]
        
        self.TARGET_SIZE = 50
        with open('/home/ubuntu/antarctica/antarctica/data/HOG_ocr_model.pkl', 'rb') as f:
            self.OCR_model = pickle.loads(f.read())
        
    def get_ocr_tool_name(self):
        return self.tool.get_name()

    def find_text(self, filmstrip):
        h, w = filmstrip.shape

        p = 0.1
        x11 = int(0.10*w - p*w)
        x12 = int(0.13*w + p*w)
        x21 = int(0.78*w - p*w)
        x22 = int(0.81*w + p*w)
        
        self._find_text(filmstrip, (x11,x12))
        self._find_text(filmstrip, (x21,x22))

    def _hog_OCR(self, char):
        char = cv2.resize(char,
                          (self.TARGET_SIZE, self.TARGET_SIZE),
                          cv2.INTER_CUBIC)

        features = hog(char).reshape(1,-1)
        prediction = self.OCR_model.predict(features)
        prediction = str(prediction[0])
        
        return prediction

    def _find_text(self, filmstrip, roi):
        h, w = filmstrip.shape
        x1, x2 = roi
           
        # perform text detection
        edges = cv2.Canny(filmstrip[:, x1:x2].astype(np.uint8), 100, 200)
        
        edges = np.amax(edges, axis=1)
        edges = scipy.ndimage.filters.median_filter(edges, 50) 
        edges = scipy.ndimage.filters.maximum_filter1d(edges, 100) 
        edges[0] = 0
        edges[h-1] = 0

        edge_locs = np.nonzero(edges[:-1] - edges[1:])[0]
        assert( len(edge_locs) % 2 == 0)

        for i in range(0, len(edge_locs), 2):
            y1 = edge_locs[i]
            y2 = edge_locs[i+1]
            length = y2 - y1
            
            # perform recognition
            segment = filmstrip[y1:y2, x1:x2].astype(np.uint8)
            segment = np.flip(np.transpose(segment), 1)

            segment = cv2.fastNlMeansDenoising(segment)
            
            char_boxes = self.tool.image_to_string(
                Image.fromarray(segment),
                lang='glacierdigits3',
                builder=pyocr.tesseract.CharBoxBuilder()
            )

            # record and label film strip
            for box in char_boxes:
                (char_y2, char_x2), (char_y1, char_x1) = box.position

                char_y1 = length - char_y1
                char_y2 = length - char_y2
                char_length = char_y2 - char_y1
                                
                if(char_length < 10 or char_length > 50):
                    self.logger.debug('invalid char dims; skipping')
                    continue

                hog_recognition = self._hog_OCR(segment[char_x2:char_x1, (length-char_y2):(length-char_y1)].astype(np.uint8))
                if(box.content != hog_recognition):
                    self.logger.debug('mismatch between tesseract and hog', box.content, hog_recognition)

                    if(box.content in ['0', '8'] and
                       hog_recognition in ['0', '8']):
                        box.content = hog_recognition

                    if(box.content in ['1', '7'] and
                       hog_recognition in ['1', '7']):
                        box.content = hog_recognition
                
                filmstrip = cv2.rectangle(filmstrip, (x1, y1+char_y1), (x2, y1+char_y2), (0,0,0))

                text = '?'
                if not box.content.isspace():
                    text = box.content
                    
                filmstrip = cv2.putText(filmstrip, text, (x1, y1+char_y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3,  cv2.LINE_AA)

        return filmstrip
        #cv2.imwrite('/home/ubuntu/test.png', filmstrip)
        
class BasicFilmstripStitcher:
    def __init__(self, logger):
        self.logger = logger
        self.images = []
            
    def _align(self, first, second):
        '''
        Use template matching to align/stitch together two images. A patch from the
        bottom of the first image will be matched to the top of the second image.
        
        Take a strip from the bottom 5% of the image from the first image and try to 
        match it to a location in the top 15% of the second image.
        '''
        
        # get a patch from the "bottom" of the first image
        h1, w1 = first.shape
        y1 = int(0.90*h1)
        y2 = int(0.95*h1)
        x1 = int(0.05*w1)
        x2 = int(0.95*w1)
        width = x2 - x1
        height = y2 - y1

        patch = first[y1:y2, x1:x2]

        # try to match the patch to the "top" of the second image
        h2, w2 = second.shape
        t1 = int(0.15*h2)

        res = cv2.matchTemplate(second[:t1,:], patch, cv2.TM_CCOEFF)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        match_x, match_y = max_loc
        if( abs(match_x - x1) > 5 ):
            self.logger.debug('Horizontal alignment off by {}. Using default offset values (120, 110)'.format(match_x - x1))
            max_loc = (120, 110)
            
        # return the alignment so that stitching can be performed
        alignment = {
            'first_bottom_margin' : h1 - y2,
            'second_top_margin' : match_y,
            'overlap' : height,
            'xoffset' : match_x - x1
        }

        return alignment

    def _stitch(self, first, second, alignment):
        '''
        Stitch together two images using their alignment. Average the overlap region.
        '''

        h1, w1 = first.shape
        h2, w2 = second.shape
        assert(w1 == w2)
        
        # assume no offset for now
        height = h1-alignment['first_bottom_margin'] + h2-alignment['second_top_margin'] - alignment['overlap']
        width = w1
        img = np.zeros( (height, width) )

        # copy in top section
        img[:h1-alignment['first_bottom_margin']-alignment['overlap'], :] = \
            first[:h1-alignment['first_bottom_margin']-alignment['overlap'], :]

        # copy in overlap region
        overlap1 = h1-alignment['first_bottom_margin']-alignment['overlap']
        overlap2 = h1-alignment['first_bottom_margin']

        img[overlap1:overlap2, :] = \
            (
             (first[overlap1:overlap2, :].astype(np.uint16) +
             second[alignment['second_top_margin']:alignment['second_top_margin']+alignment['overlap'],:].astype(np.uint16)) / 2
            ).astype(np.uint8)
        
        # copy in bottom section
        img[h1-alignment['first_bottom_margin']:, :] = \
            second[alignment['second_top_margin']+alignment['overlap']:, :]

        return img

    def stitch(self, images):

        for image in images:
            image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
            self.images.append(image)

        self.logger.info('Loaded images')

        alignments = []
        for i in range(len(self.images)-1):
            first = self.images[i]
            second = self.images[i+1]
            alignments.append( self._align(first, second) )
        
        self.logger.info('Finsihed alignment')

        stitched_image = self.images[0]
        for i in range(1, len(self.images)):
            image = self.images[i]
            alignment = alignments[i-1]
            
            stitched_image = self._stitch(stitched_image, image, alignment)
                    
        self.logger.info('Finished stitching images')
        #cv2.imwrite('/home/ubuntu/test.png', stitched_image)
        return stitched_image
