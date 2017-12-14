#!/usr/bin/env python3 

import argparse
import cv2
import errno
import itertools
import git
import os
import simplejson
import datetime

import pathos.multiprocessing as mp
import numpy as np
import AntarcticaUtils as a

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# number of frames to stitch together
BATCH_SIZE = 25
RESULTS_FILENAME = 'stitch.csv'
RESULTS_IMAGE_DIRNAME = 'stitched_images'

ZFILL_PADDING = 6
OUTPUT_FORMAT = '{reel_num},{begin_image_num},{end_image_num},{width},{height},{was_success},{error_message}'
OUTPUT_IMAGE_FILENAME_FORMAT = '{}_{}_{}-reel_begin_end.tiff'

class image_processer:

    
    def __init__(self, reel_num, output_dir):

        self.output_dir = output_dir
        self.reel_num = reel_num
        
    def get_filename_num(self, filename):

        return filename.split('_')[-1].split('.')[0]

    def __call__(self, images):

        logger.debug('processing images in reel {}'.format(self.reel_num))

        image_data = {
            'images' : images,
            'nums' : [],
            'min' : None,
            'max' : None,
            'rasters' : []
            }
            
        for image in images:
            num = self.get_filename_num(image)
            image_data['nums'].append(num)

        range_min = min(image_data['nums'])
        range_max = max(image_data['nums'])
        image_data['min'] = range_min
        image_data['max'] = range_max
        logger.debug('processing images {} to {}'.format(image_data['min'], image_data['max']))

        ########################################################################
        # attempt to open the images
        ########################################################################
        image_data['nums'] = []
        image_data['rasters'] = []
        for image in images:
            num = self.get_filename_num(image)
            image_data['nums'].append(int(num))
            
            raster = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            assert raster is not None, 'Could not open {}. Skipping batch!'.format(image)
            
            image_data['rasters'].append(raster)

        logger.debug('Loaded images')
                
        print(image_data['nums'])
        # check the size of the batch
        if len(image_data['rasters']) != BATCH_SIZE:
            logger.warning('Recieved {} frames when {} were expected. Skipping batch!'.format(len(images), BATCH_SIZE))

            assert len(image_data['rasters']) == len(image_data['images'])  

            lines = []
            for i in range(len(image_data['rasters'])):
                raster = image_data['rasters'][i]
                begin, end = str(image_data['nums'][i]).zfill(6), str(image_data['nums'][i]).zfill(ZFILL_PADDING)
                
                tiff_filename = os.path.join(self.output_dir, RESULTS_IMAGE_DIRNAME, OUTPUT_IMAGE_FILENAME_FORMAT.format(self.reel_num, begin, end))
                ret = cv2.imwrite(tiff_filename, raster)
                lines.append(OUTPUT_FORMAT.format(**{
                    'reel_num' : self.reel_num,
                    'begin_image_num' : begin,
                    'end_image_num' : end,
                    'width' : raster.shape[0],
                    'height' : raster.shape[1],
                    'was_success' : 'False',
                    'error_message' : 'not_enough_input_images'
                    }))

            return lines
            
        ########################################################################
        # try to stitch the frames together
        ########################################################################
        stitched_image = a.BasicFilmstripStitcher.stitch(image_data['rasters'], logger)

        logger.debug('Finished stitching image')
            
        ########################################################################
        # dump output to tiff file
        ########################################################################
        begin, end = str(image_data['min']).zfill(ZFILL_PADDING), str(image_data['max']).zfill(ZFILL_PADDING)
        tiff_filename = os.path.join(self.output_dir, RESULTS_IMAGE_DIRNAME, OUTPUT_IMAGE_FILENAME_FORMAT.format(self.reel_num, begin, end))
        ret = cv2.imwrite(tiff_filename, stitched_image)
        
        line = OUTPUT_FORMAT.format(**{
            'reel_num' : self.reel_num,
            'begin_image_num' : begin,
            'end_image_num' : end,
            'width' : stitched_image.shape[0],
            'height' : stitched_image.shape[1],
            'was_success' : 'True',
            'error_message' : ''
            })

        return [line]
            
def main(args):

    logger.info('Using {} worker(s)'.format(args.num_workers))
    pool = mp.Pool(args.num_workers)

    logger.debug('Checking if {} exists'.format(args.output_dir))
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            logger.error('{} already exists!'.format(args.output_dir))
            return 
        else:
            raise e

    args.output_csv = os.path.join(args.output_dir, RESULTS_FILENAME)
    args.output_image_dir = os.path.join(args.output_dir, RESULTS_IMAGE_DIRNAME)
    os.makedirs(args.output_image_dir)
        
    logger.debug('Checking if {} exists'.format(args.output_csv))
    with open(args.output_csv, 'x') as f:
        header = OUTPUT_FORMAT.replace('{', '').replace('}', '').strip()
        f.write(header+'\n')
        
    logger.info('Initialization completed')
    logger.debug('Preparing macro batches for processing')    
    macro_batch_size = BATCH_SIZE * args.num_workers
    macro_batches = []
    for i in range(0, len(args.images), macro_batch_size):
        macro_batches.append(args.images[i:i+macro_batch_size])
        
    logger.info('Batches preapred (begin processing)')
    logger.debug('Processing data in %d batch(es)' % len(macro_batches))
    logger.debug('Begin processing batch(es)')
    for macro_batch in macro_batches:
        batches = []
        for i in range(0, len(macro_batch), BATCH_SIZE):
            batches.append(macro_batch[i:i+BATCH_SIZE])

        CSVrows = itertools.chain.from_iterable(
            pool.map(image_processer(args.reel_num, args.output_dir), batches)
        )
        
        with open(args.output_csv, 'a') as f:
            f.write('\n'.join(CSVrows)+'\n')

        logger.debug('Completed a macro batch')

    pool.close()
    pool.join()
    logger.debug('Finished processing batch(es)')
    logger.info('Done!')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform OCR on radiometric data from Antarctic glaciers.')

    parser.add_argument('--parallel',
                        dest='num_workers',
                        metavar='num_workers',
                        type=int,
                        help='num worker processes to use (default: num cores on machine)',
                        default=os.cpu_count())

    parser.add_argument('--reel',
                        dest='reel_num',
                        metavar='reel_num',
                        type=str,
                        help='the reel number to use in the csv output')

    parser.add_argument('--output_dir',
                        dest='output_dir',
                        metavar='output_dir',
                        type=str,
                        help='the directory to output processed images and CSV results')

    parser.add_argument(dest='images',
                        metavar='image',
                        type=str,
                        nargs='+',
                        help='images to process (NOTE: processed in the order they appear)')

    args = parser.parse_args()
        
    main(args)
