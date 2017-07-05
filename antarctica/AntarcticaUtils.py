import cv2
import pyocr
import pyocr.builders
import numpy as np
import scipy.ndimage

from matplotlib import pyplot as plt
from PIL import Image

class BasicOCRReader:
    def __init__(self):
        self.tool = pyocr.get_available_tools()[0]
        print('Using', self.tool.get_name())

    def find_text(self, filmstrip):
        h, w = filmstrip.shape

        p = 0.01
        x11 = int(0.10*w - p*w)
        x12 = int(0.13*w + p*w)
        x21 = int(0.78*w - p*w)
        x22 = int(0.81*w + p*w)
        
        self._find_text(filmstrip, (x11,x12))
        self._find_text(filmstrip, (x21,x22))
        
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

            word_boxes = self.tool.image_to_string(
                Image.fromarray(segment),
                lang='glacierdigits3',
                builder=pyocr.builders.DigitLineBoxBuilder()
                #builder=pyocr.builders.LineBoxBuilder()
            )

            # record and label film strip
            for box in word_boxes:
                (word_y2, _), (word_y1, _) = box.position

                word_y1 = length - word_y1
                word_y2 = length - word_y2
                word_length = word_y2 - word_y1
                
                filmstrip = cv2.rectangle(filmstrip, (x1, y1+word_y1), (x2, y1+word_y2), (0,0,0))

                text = '???'
                if not box.content.isspace():
                    text = box.content
                #print(text)
                    
                filmstrip = cv2.putText(filmstrip, text, (x1, y1+word_y2+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3,  cv2.LINE_AA)
                
        cv2.imwrite('/home/ubuntu/test.png', filmstrip)
        
    def visualize(self):
        pass
    
class BasicFilmstripStitcher:
    def __init__(self):
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
            print('Warning: horizontal alignment off by', match_x - x1)
            print('Warning: using default offset values (120, 110)')
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

    def stitch(self, image_filenames):

        for image_filename in image_filenames:
            image = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
            image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
            self.images.append(image)

        print('Loaded images')

        alignments = []
        for i in range(len(self.images)-1):
            first = self.images[i]
            second = self.images[i+1]
            alignments.append( self._align(first, second) )
        
        print('Finsihed alignment')

        # print out statistics on the alignment
        #for alignment in alignments:
        #    print(alignment['first_bottom_margin'])
        #    print(alignment['second_top_margin'])
        #    print(alignment['xoffset'])
            
        stitched_image = self.images[0]
        for i in range(1, len(self.images)):
            image = self.images[i]
            alignment = alignments[i-1]
            
            stitched_image = self._stitch(stitched_image, image, alignment)
                    
        print('Finished stitching images')
        #cv2.imwrite('/home/ubuntu/test.png', stitched_image)
        return stitched_image
