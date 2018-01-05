import collections
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
        
        with open('/home/jemmons/projects/antarctica/antarctica/data/HOG_ocr_model.pkl', 'rb') as f:
            self.OCR_model = pickle.loads(f.read())

        numbers = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png']
        self.num_templates = []
        for number in numbers:
            template = cv2.imread(os.path.join('/home/jemmons/projects/antarctica/antarctica/data/', number), cv2.IMREAD_GRAYSCALE)
            self.num_templates.append(template.astype(np.uint8))
        self.number_height = 50
        self.number_padding = 40

        backwards_numbers = ['3.png', '4.png', '6.png', '7.png', '9.png']
        self.backwards_num_templates = []
        for number in backwards_numbers:
            template = cv2.imread(os.path.join('/home/jemmons/projects/antarctica/antarctica/data/', number), cv2.IMREAD_GRAYSCALE)
            template = cv2.flip(template, 1)
            self.backwards_num_templates.append(template.astype(np.uint8))

        self.number_spacing = 1250
        
    def get_ocr_tool_name(self):

        return self.tool.get_name()

    def orient(self, filmstrip, logger):
        return filmstrip


        #TODO(jremmons) make this work...
        h, w = filmstrip.shape

        logger.debug('orienting the filmstrip')
        
        p = 0.05
        y11 = int(0)
        y12 = int(0.25*h)
        y21 = int(0.65*h)
        y22 = int(h)

        # perform a heuristic to fix the orientation (look for backwars digits)
        top_line = (filmstrip[y11:y12, :] * (255 / 65535.0)).astype(np.uint8)
        bottom_line = (filmstrip[y21:y22, :] * (255 / 65535.0)).astype(np.uint8)

        top_thres = cv2.adaptiveThreshold(top_line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 15)
        bottom_thres = cv2.adaptiveThreshold(bottom_line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 15)

        flip_evidence = 0
        for template in self.backwards_num_templates:
            top_res = cv2.matchTemplate(top_thres, template, cv2.TM_CCOEFF_NORMED)
            top_y, top_x = np.where(top_res >= 0.8)
            if top_y.any():
                flip_evidence += 1
            
            bottom_res = cv2.matchTemplate(bottom_thres, template, cv2.TM_CCOEFF_NORMED)
            bottom_y, bottom_x = np.where(bottom_res >= 0.8)
            if bottom_y.any():
                flip_evidence += 1

        cv2.imwrite('/home/jemmons/im.jpg', (filmstrip[:, :] * (255 / 65535.0)).astype(np.uint8))
        if flip_evidence > 4:
            logger.critical('flipped the flipstrip!')
            filmstrip = v2.flip(filmstrip.copy(), 1)
                
        return filmstrip
    
    def find_text(self, oriented_filmstrip, preprocessing_metadata, logger):

        h, w = oriented_filmstrip.shape
       
        # y11 = int(0)
        # y12 = int(0.25*h)
        # y21 = int(0.65*h)
        # y22 = int(h)

        # top_line = (oriented_filmstrip[y11:y12, :] * (255 / 65535.0)).astype(np.uint8)
        # bottom_line = (oriented_filmstrip[y21:y22, :] * (255 / 65535.0)).astype(np.uint8)

        # cv2.imwrite('/home/jemmons/roi1.png', top_line)
        # cv2.imwrite('/home/jemmons/roi2.png', bottom_line)

        # top_thres = cv2.adaptiveThreshold(top_line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 15)
        # bottom_thres = cv2.adaptiveThreshold(bottom_line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 15)

        # cv2.imwrite('/home/jemmons/roi1.png', top_thres)
        # cv2.imwrite('/home/jemmons/roi2.png', bottom_thres)

        # top_yres = []
        # bottom_yres = []
        # for template in self.num_templates:
        #     top_res = cv2.matchTemplate(top_thres, template, cv2.TM_CCOEFF_NORMED)
        #     top_y, top_x = np.where(top_res >= 0.8)
        #     if top_y.any():
        #         top_yres.append(np.median(top_y))
            
        #     bottom_res = cv2.matchTemplate(bottom_thres, template, cv2.TM_CCOEFF_NORMED)
        #     bottom_y, bottom_x = np.where(bottom_res >= 0.8)
        #     if bottom_y.any():
        #         bottom_yres.append(np.median(bottom_y))
        #     # TODO(jremmons) add a check so that uncertainty in the number position is considered
                
        # top_centerline = int(np.average(top_yres) + self.number_height/2)
        # bottom_centerline = int(np.average(bottom_yres) + self.number_height/2)

        # top_line = cv2.rectangle(top_line.copy(), (0, top_centerline-self.number_padding), (w, top_centerline+self.number_padding), (0,0,0), thickness=5)
        # bottom_line = cv2.rectangle(bottom_line.copy(), (0, bottom_centerline-self.number_padding), (w, bottom_centerline+self.number_padding), (0,0,0), thickness=5)
        # cv2.imwrite('/home/jemmons/roi1.png', top_line)
        # cv2.imwrite('/home/jemmons/roi2.png', bottom_line)

        # assert False
        
        # top_y1 = y11+top_centerline-self.number_padding
        # top_y2 = y11+top_centerline+self.number_padding
        # top_textline = oriented_filmstrip[top_y1:top_y2, :]

        # bottom_y1 = y21+bottom_centerline-self.number_padding
        # bottom_y2 = y21+bottom_centerline+self.number_padding
        # bottom_textline = oriented_filmstrip[bottom_y1:bottom_y2, :]

        top_detections_final = []
        if preprocessing_metadata['top_line']:
            ymin, ymax = preprocessing_metadata['top_line_y']
            ymin = int(ymin)
            ymax = int(ymax)
            
            top_textline = oriented_filmstrip[ymin:ymax,:]
            top_detections = self._recognize_text(top_textline, logger)

            for d in top_detections:
                d['xmin'] = str(d['x1'])
                d['xmax'] = str(d['x2'])
                d['ymin'] = str(d['y1'] + ymin)
                d['ymax'] = str(d['y2'] + ymin)
                d['recognition_type'] = 'number_char'
                top_detections_final.append(d)

        bottom_detections_final = []
        if preprocessing_metadata['bot_line']:
            ymin, ymax = preprocessing_metadata['bot_line_y']
            ymin = int(ymin)
            ymax = int(ymax)

            bot_textline = oriented_filmstrip[ymin:ymax,:]
            bottom_detections = self._recognize_text(bottom_textline, logger)

            for d in bottom_detections:
                d['xmin'] = str(d['x1'])
                d['xmax'] = str(d['x2'])
                d['ymin'] = str(d['y1'] + ymin)
                d['ymax'] = str(d['y2'] + ymin)
                d['recognition_type'] = 'number_char'
                bottom_detections_final.append(d)

        return top_detections_final, bottom_detections_final

    @staticmethod
    def annotate_image(image, number_groups, logger):
        image = image.copy()
        
        logger.debug('Adding char labels to image')
            
        for number_group in number_groups:
            if number_group['recognition_type'] != 'number_group':
                logger.debug('Skipping number')
                continue

            chars = number_group['chars']

            xmin = int(number_group['xmin'])
            xmax = int(number_group['xmax'])
            ymin = int(number_group['ymin'])
            ymax = int(number_group['ymax'])
            image = cv2.rectangle(image, (xmin-3, ymin-3), (xmax+3, ymax+3), (0,0,0), thickness=6)
            image = cv2.putText(image, chars, (xmin, ymax+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3,  cv2.LINE_AA)

        return image

    @staticmethod
    def annotate_image_digits(image, number_groups, logger):
        image = image.copy()
        
        logger.debug('Adding char labels to image')
            
        for number_group in number_groups:
            if number_group['recognition_type'] != 'number_char':
                logger.debug('Skipping number')
                continue

            chars = number_group['char']

            xmin = int(number_group['xmin'])
            xmax = int(number_group['xmax'])
            ymin = int(number_group['ymin'])
            ymax = int(number_group['ymax'])
            image = cv2.rectangle(image, (xmin-3, ymin-3), (xmax+3, ymax+3), (0,0,0), thickness=6)
            image = cv2.putText(image, chars, (xmin, ymax+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3,  cv2.LINE_AA)

        return image


    def _interpret_text(self, text_detections, shape, logger):
        top_detections, bottom_detections = text_detections
        h, w = shape
        
        ################################################################################ 
        # gather statistics about the detected characters
        ################################################################################ 
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
        
        ################################################################################ 
        # parse the digits
        ################################################################################ 
        number_groups = []
        def parse_grouping(group, interval, t):

            xmin = group[interval[0]]['xmin']
            xmax = group[interval[0]]['xmax']
            ymin = group[interval[0]]['ymin']
            ymax = group[interval[0]]['ymax']
            
            digits = ''
            for i in range(interval[0], interval[1]):                
                group[i]['number_type'] = t

                if group[i]['xmin'] < xmin:
                    xmin = group[i]['xmin']
                if group[i]['xmax'] > xmax:
                    xmax = group[i]['xmax']
                if group[i]['ymin'] < ymin:
                    ymin = group[i]['ymin']
                if group[i]['ymax'] > ymax:
                    ymax = group[i]['ymax']

                digits += group[i]['char']
                
            number_group = {'xmin' : str(xmin),
                            'xmax' : str(xmax),
                            'ymin' : str(ymin),
                            'ymax' : str(ymax),
                            'number_type' : t,
                            'chars' : digits,
                            'recognition_type' : 'number_group'}
            
            number_groups.append(number_group)
            return digits

        # TODO(jremmons) handle missing digit cases
        
        # perform an intelligent recognition on the top strip
        # should be the date; expressed as two 6-digit numbers
        date_numbers = []
        time_numbers = []
        top_successes = 0
        for group in top_groups:
            if(len(group) != 12):
                logger.debug('incorrect number of digits in the top line: 12!={}'.format(len(group)))
                date_numbers.append(None)
                time_numbers.append(None)
                continue

            date_numbers.append(parse_grouping(group, (0,6),'date'))
            time_numbers.append(parse_grouping(group, (6,12),'time'))
            
            top_successes += 1

        # perform an intelligent recognition on the bottom strip
        # should be a 4-digit setting, 3-digit flight, and 6-digit cbd
        setting_numbers = []
        flight_numbers = []
        cbd_numbers = []
        bottom_successes = 0
        for group in bottom_groups:
            if(len(group) != 11):
                logger.debug('incorrect number of digits in the bottom line: 11!={}'.format(len(group)))
                setting_numbers.append(None)
                flight_numbers.append(None)
                cbd_numbers.append(None)
                continue

            setting_numbers.append(parse_grouping(group, (0,4),'setting'))
            flight_numbers.append(parse_grouping(group, (4,7),'flight'))
            cbd_numbers.append(parse_grouping(group, (7,11),'cbd'))
            bottom_successes += 1

        ################################################################################ 
        # perform sanity check to ensure enough digits were parsed from the film
        ################################################################################ 

        #expected_groups = int(w / (self.number_spacing + 850))
        if top_successes < 5:
            logger.warning('too many errors (< 5 good detections: {}) when parsing the top line; skipping batch!'.format(top_successes))
            return None

        if bottom_successes < 5:
            logger.warning('too many errors (< 5 good detections: {}) when parsing the bottom line; skipping batch!'.format(bottom_successes))
            return None

        ################################################################################ 
        # perform semantic sanity check on the "constant" digits from each segment
        ################################################################################ 

        def interpert_constants(numbers):
            counter = collections.Counter()
            for number in numbers:
                counter[number] += 1

            ret = counter.most_common(2)
            if len(ret) == 2:
                most_common, second_most_common = counter.most_common(2)
                return most_common, second_most_common
            else:
                return ret[0], ('NaN', 0)
            
        date_most_common, date_second_most_common = interpert_constants(date_numbers)
        if date_most_common[1] < 2*date_second_most_common[1]:
            logger.warning('too many errors (>25%) when interperting the date; skipping batch!')
            return None
        date_number_final = date_most_common[0]
        
        setting_most_common, setting_second_most_common = interpert_constants(setting_numbers)
        if setting_most_common[1] < 2*setting_second_most_common[1]:
            logger.warning('too many errors (>25%) when interperting the setting; skipping batch!')
            return None
        setting_number_final = setting_most_common[0]
        
        flight_most_common, flight_second_most_common = interpert_constants(flight_numbers)
        if flight_most_common[1] < 2*flight_second_most_common[1]:
            logger.warning('too many errors (>25%) when interperting the flight; skipping batch!')
            return None
        flight_number_final = flight_most_common[0]
        
        ################################################################################ 
        # perform indepth sanity check on cbds
        ################################################################################ 
        cbd_deltas = []
        for i in range(1, len(cbd_numbers)):
            if cbd_numbers[i-1] is not None and cbd_numbers[i] is not None:
                cbd_deltas.append(int(cbd_numbers[i]) - int(cbd_numbers[i-1]))

        cbd_delta_most_common, cbd_delta_second_most_common = interpert_constants(cbd_deltas)
        if cbd_delta_most_common[1] < 2*cbd_delta_second_most_common[1]:
            logger.warning('too many errors (>25%) when interperting the cbd_delta (cbd:count, {}:{}, {}:{}); skipping batch!'.format(cbd_delta_most_common[0], cbd_delta_most_common[1], cbd_delta_second_most_common[0], cbd_delta_second_most_common[1]))
            return None

        if cbd_delta_most_common[0] != 1 and cbd_delta_most_common[0] != -1: 
            logger.warning('invalid cbd_delta (should be -1 or 1, but it was {}); skipping batch!'.format(cbd_delta_most_common[0]))
            return None
        
        cbd_delta_number_final = cbd_delta_most_common[0]

        # fill in the missing/incorrectly recognized CBDs
        # TODO(jremmons) confirm that the edits made are correct (i.e. check the residual)
        cbd_fix = []
        for i in range(1, len(cbd_numbers)):
            if cbd_numbers[i-1] is not None and cbd_numbers[i] is not None:
                if int(cbd_numbers[i]) - int(cbd_numbers[i-1]) == cbd_delta_number_final:
                    cbd_fix.append(int(cbd_numbers[i-1]))
                    continue
            cbd_fix.append(None)

        valid_cbd1_num = str(list(filter(None, cbd_fix))[0]).zfill(4)
        valid_cbd2_num = str(list(filter(None, cbd_fix))[-1]).zfill(4)
        
        cbds_numbers = list(filter(lambda x: x['number_type'] == 'cbd', number_groups))

        valid_cbd1 = list(filter(lambda x: x['chars'] == valid_cbd1_num, cbds_numbers))
        valid_cbd2 = list(filter(lambda x: x['chars'] == valid_cbd2_num, cbds_numbers))
        if len(valid_cbd1) > 1:
            logger.warning('multiple cbds with the same value detected: cbd={} count={}'.format(valid_cbd1_num, len(valid_cbd1)))

        if len(valid_cbd2) > 1:
            logger.warning('multiple cbds with the same value detected: cbd={} count={}'.format(valid_cbd2_num, len(valid_cbd2)))
            
        valid_cbd1 = valid_cbd1[0]
        valid_cbd2 = valid_cbd2[-1]

        cbd1_fractional_mid = (int(valid_cbd1['xmin']) + int(valid_cbd1['xmax'])) / 2
        cbd2_fractional_mid = (int(valid_cbd2['xmin']) + int(valid_cbd2['xmax'])) / 2
        cbd_grad = (int(valid_cbd1['chars']) - int(valid_cbd2['chars'])) / (cbd1_fractional_mid - cbd2_fractional_mid)
        
        cbd1_fractional = cbd_grad * (0 - cbd1_fractional_mid) + int(valid_cbd1['chars'])
        cbd2_fractional = cbd_grad * (w - cbd2_fractional_mid) + int(valid_cbd2['chars'])
        
        cbd_first_idx = 0
        for i in range(len(cbd_fix)):
            if cbd_fix[i] is not None:
                cbd_first_idx = i
                break

        for i in range(len(cbd_fix)):
            if cbd_fix[i] is None:
                cbd_fix[i] = cbd_fix[cbd_first_idx] + cbd_delta_number_final * (i - cbd_first_idx)                    

        cbd_final = list(map(lambda x : str(x).zfill(4), cbd_fix))
 
        ################################################################################ 
        # perform indepth sanity check on time values
        ################################################################################ 
        def time_sanity_check(t):
            if t is None:
                return None

            if t[0] not in [ str(i) for i in range(0,3)]:
                logger.debug('the time value {} is not well-formed in digit 0 ({} not in [0-2])'.format(t, t[0]))
                return None
            if t[1] not in [ str(i) for i in range(0,10)]:
                logger.debug('the time value {} is not well-formed in digit 1 ({} not in [0-9])'.format(t, t[1]))
                return None
            if t[2] not in [ str(i) for i in range(0,6)]:
                logger.debug('the time value {} is not well-formed in digit 2 ({} not in [0-6])'.format(t, t[2]))
                return None
            if t[3] not in [ str(i) for i in range(0,10)]:
                logger.debug('the time value {} is not well-formed in digit 3 ({} not in [0-9])'.format(t, t[3]))
                return None
            if t[4] not in [ str(i) for i in range(0,6)]:
                logger.debug('the time value {} is not well-formed in digit 4 ({} not in [0-5])'.format(t, t[4]))
                return None
            if t[5] not in [ str(i) for i in range(0,10)]:
                logger.debug('the time value {} is not well-formed in digit 5 ({} not in [0-9])'.format(t, t[5]))
                return None

            return t
            
        def time2sec(t):
            if t is None:
                return None
            
            hour = int(t[0:2])
            minute = int(t[2:4])
            second = int(t[4:6])

            return 3600*hour + 60*minute + second
            
        def sec2time(n):
            if n is None:
                return None

            hour = n // 3600
            n -= 3600 * hour

            minute = n // 60
            n -= 60*minute

            second = n
            return str(hour).zfill(2) + str(minute).zfill(2) + str(second).zfill(2)
            
        # fix the missing/incorrectly recognized time values        
        time_numbers_seconds = list(map(time2sec, map(time_sanity_check, time_numbers)))

        # TODO(jremmons) make sure there are enough time values to 

        time_deltas = []
        for i in range(1, len(time_numbers_seconds)):
            if time_numbers_seconds[i-1] is not None and time_numbers_seconds[i] is not None:
                time_deltas.append(int(time_numbers_seconds[i]) - int(time_numbers_seconds[i-1]))

        time_delta_most_common, time_delta_second_most_common = interpert_constants(time_deltas)
        if time_delta_most_common[1] < 2*time_delta_second_most_common[1]:
            logger.warning('too many errors (>25%) when interperting the time_delta; skipping batch!')
            return None

        if time_delta_most_common[0] != 15 and time_delta_most_common[0] != -15: 
            logger.warning('invalid time_delta (should be -15 or 15, but it was {}); skipping batch!'.format(time_delta_most_common[0]))
            return None
        
        time_delta_number_final = time_delta_most_common[0]

        # fill in the missing/incorrectly recognized TIMEs
        # TODO(jremmons) confirm that the edits made are correct (i.e. check the residual)
        time_fix = []
        for i in range(1, len(time_numbers_seconds)):
            if time_numbers_seconds[i-1] is not None and time_numbers_seconds[i] is not None:
                if int(time_numbers_seconds[i]) - int(time_numbers_seconds[i-1]) == time_delta_number_final:
                    time_fix.append(int(time_numbers_seconds[i-1]))
                    continue
            time_fix.append(None)

        time_first_idx = 0
        for i in range(len(time_fix)):
            if time_fix[i] is not None:
                time_first_idx = i
                break

        for i in range(len(time_fix)):
            if time_fix[i] is None:
                time_fix[i] = time_fix[time_first_idx] + time_delta_number_final * (i - time_first_idx)
                
        time_final = list(map(str, map(sec2time, time_fix)))

        ################################################################################ 
        # gather the data needed for the interpretation
        ################################################################################ 
        interpretation = {
            'date' : date_number_final,
            'time1' : time_final[0],
            'time2' : time_final[-1],
            'setting' : setting_number_final,
            'flight' : flight_number_final,
            'cbd1' : cbd_final[0],
            'cbd2' : cbd_final[-1],
            'cbd1_fractional' : cbd1_fractional,
            'cbd2_fractional' : cbd2_fractional
        }
        
        return interpretation, number_groups
            
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
                
                # if(char_length < 10 or char_length > 50 or
                #    char_height < 10 or char_height > 100):
                #     logger.debug('invalid char dims; skipping! (I thought it might be: {})'.format(box.content))
                #     continue
                
                hog_recognition = self._hog_OCR(segment[char_y1:char_y2, char_x1:char_x2].astype(np.uint8), logger)
                if(box.content != hog_recognition):
                    logger.debug('mismatch between tesseract and hog: ' + str(box.content) + ' ' + str(hog_recognition))

                    if(box.content in ['0', '8'] and hog_recognition in ['0', '8']):
                        box.content = hog_recognition
                        
                    if(box.content in ['1', '7'] and hog_recognition in ['1', '7']):
                        box.content = hog_recognition
                

                text = '?'
                if box.content in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    text = box.content

                logger.debug('add rectangles around the characters')
                textline = cv2.rectangle(textline.copy(), (x1+char_x1, char_y1), (x1+char_x2, char_y2), (0,0,0))                
                textline = cv2.putText(textline.copy(), text, (x1+char_x1+20, char_y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3,  cv2.LINE_AA)

                detection = {'char' : text,
                             'x1' : x1+char_x1,
                             'x2' : x1+char_x2,
                             'y1' : char_y1,
                             'y2' : char_y2,
                             'segment': segment_number}

                text_detections.append(detection)
            segment_number += 1
                
        cv2.imwrite('/home/jemmons/roi1.png', textline)
        return text_detections
        
class BasicFilmstripStitcher:

    @staticmethod
    def stitch(uint16_images, logger):

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

            alignment = BasicFilmstripStitcher._align(first, second, logger)
            if alignment is None:
                return None

            alignments.append(alignment)
                
        if logger:
            logger.debug('Finsihed alignment')

        stitched_image = uint16_images[0]
        for i in range(1, len(uint16_images)):
            uint16_image = uint16_images[i]
            alignment = alignments[i-1]

            stitched_image = BasicFilmstripStitcher._stitch(stitched_image, uint16_image, alignment, logger)
            
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
                logger.error('Horizontal alignment off by {}. Using default offset values (x1=120, x2=110, y=0)'.format(match_x - x1))
                logger.warning('Horizontal alignment off by {}. Skipping batch'.format(match_x - x1))
            #return None
            match_x, match_y = (120, 110) # TODO(jremmons) default offset
            
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
