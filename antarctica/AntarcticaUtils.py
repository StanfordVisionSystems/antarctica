import cv2
import os
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


    def __init__(self):

        self.tool = pyocr.get_available_tools()[0]
        
        with open('/home/ubuntu/antarctica/antarctica/data/HOG_ocr_model.pkl', 'rb') as f:
            self.OCR_model = pickle.loads(f.read())

        numbers = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png']
        self.num_templates = []
        for number in numbers:
            template = cv2.imread(os.path.join('/home/ubuntu/antarctica/antarctica/data/', number), cv2.IMREAD_GRAYSCALE)
            self.num_templates.append(template.astype(np.uint8))
        self.number_height = 50
        self.number_padding = 40

        self.number_spacing = 1250
        
    def get_ocr_tool_name(self):

        return self.tool.get_name()

    def orient(self, filmstrip, logger):

        h, w = filmstrip.shape

        p = 0.05
        y11 = int(0.115*h - p*h)
        y12 = int(0.115*h + p*h)
        y21 = int(0.795*h - p*h)
        y22 = int(0.795*h + p*h)

        # perform some kind of heuristic to determine the correct orientation
        
        # god only knows knows why we need the copy...
        #filmstrip = cv2.rectangle(filmstrip.copy(), (0, y11), (w, y12), (0,0,0), thickness=5)
        #filmstrip = cv2.rectangle(filmstrip.copy(), (0, y21), (w, y22), (0,0,0), thickness=5)
        
        return filmstrip
    
    def find_text(self, oriented_filmstrip, logger):

        h, w = oriented_filmstrip.shape
       
        y11 = int(0)
        y12 = int(0.25*h)
        y21 = int(0.65*h)
        y22 = int(h)

        top_line = (oriented_filmstrip[y11:y12, :] * (255 / 65535.0)).astype(np.uint8)
        bottom_line = (oriented_filmstrip[y21:y22, :] * (255 / 65535.0)).astype(np.uint8)

        top_thres = cv2.adaptiveThreshold(top_line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 15)
        bottom_thres = cv2.adaptiveThreshold(bottom_line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 15)

        top_yres = []
        bottom_yres = []
        for template in self.num_templates:
            top_res = cv2.matchTemplate(top_thres, template, cv2.TM_CCOEFF_NORMED)
            top_y, topx = np.where(top_res >= 0.8)
            if top_y.any():
                top_yres.append(np.median(top_y))
            
            bottom_res = cv2.matchTemplate(bottom_thres, template, cv2.TM_CCOEFF_NORMED)
            bottom_y, bottom_x = np.where(bottom_res >= 0.8)
            if bottom_y.any():
                bottom_yres.append(np.median(bottom_y))
            # TODO(jremmons) add a check so that uncertainty in the number position is considered
                
        top_centerline = int(np.average(top_yres) + self.number_height/2)
        bottom_centerline = int(np.average(bottom_yres) + self.number_height/2)

        '''
        #top_line = cv2.rectangle(top_line.copy(), (0, top_centerline-self.number_padding), (w, top_centerline+self.number_padding), (0,0,0), thickness=5)
        #bottom_line = cv2.rectangle(bottom_line.copy(), (0, bottom_centerline-self.number_padding), (w, bottom_centerline+self.number_padding), (0,0,0), thickness=5)
        #cv2.imwrite('/home/ubuntu/roi1.png', top_line)
        #cv2.imwrite('/home/ubuntu/roi2.png', bottom_line)
        '''
        
        top_textline = oriented_filmstrip[y11+top_centerline-self.number_padding:y11+top_centerline+self.number_padding, :]
        bottom_textline = oriented_filmstrip[y21+bottom_centerline-self.number_padding:y21+bottom_centerline+self.number_padding, :]

        '''
        #cv2.imwrite('/home/ubuntu/roi1.png', top_textline)
        #cv2.imwrite('/home/ubuntu/roi2.png', bottom_textline)
        '''
        
        top_detections = self._recognize_text(top_textline, logger)
        bottom_detections = self._recognize_text(bottom_textline, logger)

        success, interpretation = self._interpret_text(top_detections, bottom_detections)
        
        if(not success):
            return None

        return interpretation
        
    def _interpret_text(self, top_detections, bottom_detections):

        # gather statistics about the detected characters
        def stats(arr):
            median = np.median(arr)
            average = np.average(arr)
            std = np.std(arr)
            mid = median if abs(average - median) > std/2.0 else average

            return median, average, mid, std
            

        top_y1 = np.array(list(map(lambda x : x['y1'], top_detections)))
        top_y1_median, top_y1_average, top_y1_std, top_y1_mid = stats(top_y1)

        top_y2 = np.array(list(map(lambda x : x['y2'], top_detections)))
        top_y2_median, top_y2_average, top_y2_std, top_y2_mid = stats(top_y2)
        
        bottom_y1 = np.array(list(map(lambda x : x['y1'], bottom_detections)))
        bottom_y1_median, bottom_y1_average, bottom_y1_std, bottom_y1_mid = stats(bottom_y1)

        bottom_y2 = np.array(list(map(lambda x : x['y2'], bottom_detections)))
        bottom_y2_median, bottom_y2_average, bottom_y2_std, bottom_y2_mid = stats(bottom_y2)

        # split the detected characters into groups
        top_groups = []
        current_group = []
        current_x = top_detections[0]['x1']
        for detection in top_detections:
            if(detection['x1'] < current_x + self.number_spacing):
                current_x = detection['x1']
                current_group.append(detection)
            else:
                top_groups.append(current_group)

                current_x = detection['x1']
                current_group = [detection]

        bottom_groups = []
        current_group = []
        current_x = bottom_detections[0]['x1']
        for detection in bottom_detections:
            if(detection['x1'] < current_x + self.number_spacing):
                current_x = detection['x1']
                current_group.append(detection)
            else:
                bottom_groups.append(current_group)

                current_x = detection['x1']
                current_group = [detection]

        # TODO(jremmons) do something with the stats on the digit locations
                
        # perform an intelligent recognition on the top strip
        # should be the date; expressed as two 6-digit numbers
        date = []
        time = []
        top_errors = 0
        for group in top_groups:
            if(len(group) != 12):
                top_errors += 1
                continue

            group_date = ''
            for i in range(0,6):
                group_date += group[i]['char']
            date.append(int(group_date))

            group_time = ''
            for i in range(6,12):
                group_time += group[i]['char']
            time.append(int(group_time))

        print(date)
        print(time)
        
        # perform an intelligent recognition on the bottom strip
        # should be a 4-digit setting, 3-digit flight, and 6-digit cbd
        setting_number = []
        flight_number = []
        cbd_number = []
        bottom_errors = 0
        for group in bottom_groups:
            if(len(group) != 11):
                bottom_errors += 1
                continue

            group_setting = ''
            for i in range(0, 4):
                group_setting += group[i]['char']
            setting_number.append(int(group_setting))

            group_flight = ''
            for i in range(4, 7):
                group_flight += group[i]['char']
            flight_number.append(int(group_flight))

            group_cbd = ''
            for i in range(7, 11):
                group_cbd += group[i]['char']
            cbd_number.append(int(group_cbd))

        print(setting_number)
        print(flight_number)
        print(cbd_number)
        print(top_errors)
        print(bottom_errors)
        
        #TODO(jremmons) do something with this information...

        #TODO(jremmons) check that the date is constant (allow for 1 or 2 error)
        #TODO(jremmons) check that the time is incrementing in roughly the same interval each time
        #TODO(jremmons) check that the settings remain constant
        #TODO(jremmons) check that the flight number is constant
        #TODO(jremmons) check that the cbd is incrementing by the same interval each time

        return None, None
            
    def _hog_OCR(self, char, logger):

        char = cv2.resize(char,
                          (self.number_height, self.number_height),
                          cv2.INTER_CUBIC)

        features = hog(char, block_norm='L2').reshape(1,-1)
        prediction = self.OCR_model.predict(features)
        prediction = str(prediction[0])
        
        return prediction

    def _recognize_text(self, textline, logger):

        h, w = textline.shape
        uint8_textline = (textline * (255 / 65535.0)).astype(np.uint8)

        edges = cv2.Canny(uint8_textline, 100, 200)

        edges = np.amax(edges, axis=0)
        edges = scipy.ndimage.filters.median_filter(edges, 50) 
        edges = scipy.ndimage.filters.maximum_filter1d(edges, 100) 
        edges[0] = 0
        edges[w-1] = 0

        edge_locs = np.nonzero(edges[:-1] - edges[1:])[0]
        assert( len(edge_locs) % 2 == 0)

        text_detections = []        
        segment_number = 0
        for i in range(0, len(edge_locs), 2):
            x1 = edge_locs[i]
            x2 = edge_locs[i+1]
            length = x2 - x1
            
            # perform recognition
            segment = uint8_textline[:, x1:x2]

            char_boxes = self.tool.image_to_string(
                Image.fromarray(segment),
                lang='glacierdigits3',
                builder=pyocr.tesseract.CharBoxBuilder()
            )

            # record and label film strip
            for box in char_boxes:
                (char_x1, char_y1), (char_x2, char_y2) = box.position

                # the output of tesseract has the origin in the lower-left (not the usual upper-left convention)
                char_y1 = h - char_y1
                char_y2 = h - char_y2
                char_y1, char_y2 = char_y2, char_y1

                char_height = char_y2 - char_y1
                char_length = char_x2 - char_x1
                
                if(char_length < 10 or char_length > 50 or
                   char_height < 10 or char_height > 100):
                    logger.debug('invalid char dims; skipping!')
                    continue
                
                hog_recognition = self._hog_OCR(segment[char_y1:char_y2, char_x1:char_x2].astype(np.uint8), logger)
                if(box.content != hog_recognition):
                    logger.debug('mismatch between tesseract and hog: ' + str(box.content) + ' ' + str(hog_recognition))

                    if(box.content in ['0', '8'] and
                       hog_recognition in ['0', '8']):
                        box.content = hog_recognition
                        
                    if(box.content in ['1', '7'] and
                       hog_recognition in ['1', '7']):
                        box.content = hog_recognition

                '''
                #logger.debug('add rectangles around the characters')
                #textline = cv2.rectangle(textline.copy(), (x1+char_x1, char_y1), (x1+char_x2, char_y2), (0,0,0))
                '''
                
                text = '?'
                if not box.content.isspace():
                    assert(len(box.content) == 1)
                    text = box.content

                '''
                #textline = cv2.putText(textline.copy(), text, (x1+char_x1+20, char_y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3,  cv2.LINE_AA)
                '''

                detection = {'char' : text,
                             'x1' : x1+char_x1,
                             'x2' : x1+char_x2,
                             'y1' : char_y1,
                             'y2' : char_y2,
                             'segment': segment_number}

                text_detections.append(detection)
            segment_number += 1
                
        return text_detections
        
class BasicFilmstripStitcher:

    @staticmethod
    def stitch(uint16_images, logger=None):

        converted_images = []
        for uint16_image in uint16_images:
            assert(uint16_image.dtype == np.uint16)
            float_image = cv2.convertScaleAbs(uint16_image, alpha=(255.0/65535.0))
            converted_images.append(float_image)

        if logger:
            logger.debug('Finsihed image conversion to float')

        alignments = []
        for i in range(len(converted_images)-1):
            first = converted_images[i]
            second = converted_images[i+1]
            alignments.append( BasicFilmstripStitcher._align(first, second, logger) )
            
        if logger:
            logger.debug('Finsihed alignment')

        stitched_image = uint16_images[0]
        for i in range(1, len(uint16_images)):
            uint16_image = uint16_images[i]
            alignment = alignments[i-1]

            stitched_image = BasicFilmstripStitcher._stitch(stitched_image, uint16_image, alignment, logger)
            
        if logger:
            logger.debug('Finished stitching images')

        return np.flip(np.transpose(stitched_image), 1)
    
    @staticmethod
    def _align(first, second, logger):
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
            if logger:
                self.logger.error('Horizontal alignment off by {}. Using default offset values (120, 110)'.format(match_x - x1))

            return None
            #max_loc = (120, 110) # TODO(jremmons) default offset
            
        # return the alignment so that stitching can be performed
        alignment = {
            'first_bottom_margin' : h1 - y2,
            'second_top_margin' : match_y,
            'overlap' : height,
            'xoffset' : match_x - x1
        }

        return alignment

    @staticmethod
    def _stitch(first, second, alignment, logger):
        '''
        Stitch together two images using their alignment. Average the overlap region.
        '''

        h1, w1 = first.shape
        h2, w2 = second.shape
        assert(w1 == w2)
        
        # assume no offset for now
        height = h1-alignment['first_bottom_margin'] + h2-alignment['second_top_margin'] - alignment['overlap']
        width = w1
        img = np.zeros((height, width), dtype=np.uint16)

        # copy in top section
        img[:h1-alignment['first_bottom_margin']-alignment['overlap'], :] = \
            first[:h1-alignment['first_bottom_margin']-alignment['overlap'], :]

        # copy in overlap region
        overlap1 = h1-alignment['first_bottom_margin']-alignment['overlap']
        overlap2 = h1-alignment['first_bottom_margin']

        img[overlap1:overlap2, :] = \
            (
             (first[overlap1:overlap2, :].astype(np.uint32) +
             second[alignment['second_top_margin']:alignment['second_top_margin']+alignment['overlap'],:].astype(np.uint32)) / 2
            ).astype(np.uint16)
        
        # copy in bottom section
        img[h1-alignment['first_bottom_margin']:, :] = \
            second[alignment['second_top_margin']+alignment['overlap']:, :]

        return img
