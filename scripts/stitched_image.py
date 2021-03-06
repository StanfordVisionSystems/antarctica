import cv2
import numpy as np
import multiprocessing as mp
import os
import threading
import timeit

from PIL import Image

class StitchedImage:

    OUTPUT_CSV_FORMAT = '{flip_x}, {flip_y}, {top_line}, {top_line_y}, {top_num_groups}, {bot_line}, {bot_line_y}, {bot_num_groups}\n'
    
    @staticmethod
    def get_base_mode():
        return {
            'flip_x': False,
            'flip_y': False,
            'top_line' : False,
            'top_line_y' : [],
            'top_num_groups' : 0,
            'bot_line' : False,
            'bot_line_y' : [],
            'bot_num_groups' : 0,
        }

    @staticmethod
    def _load_image(image_path, scale_factor):

        image_original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        assert image_original is not None, 'failed to load image! {}'.format(image_path)

        image_uint8 = image_original
        if image_original.dtype == np.uint16: # convert to uint16 -> uint8 for display if needed
            image_uint8 = (image_original * (255 / 65535)).astype(np.uint8)

        elif image_original.dtype != np.uint8: # unknown image size; throw error
            assert image_original.dtype == np.uint8
            
        image_uint8 = Image.fromarray(image_uint8)
        w, h = image_uint8.size
        w = int(w * (1.0/scale_factor))
        h = int(h * (1.0/scale_factor))
        image_uint8 = image_uint8.resize((w, h)) 

        return {
            'image_original' : image_original, 
            'image_uint8' : image_uint8
        }

    @staticmethod
    def _commit_to_disk(image, image_basename, output_dir, scale_factor, mode):

        # preform any necessary processing
        if mode['flip_x']:
            image = cv2.flip(image, 1) # flip around the y-axis
            #image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if mode['flip_y']:
            image = cv2.flip(image, 0) # flip around the x-axis
            #image = image.transpose(Image.FLIP_TOP_BOTTOM)

        cv2.imwrite(os.path.join(output_dir, image_basename), image)

        # multiple by the scale factor
        # print(image_basename, mode)
        if mode['top_line']:
            assert mode['top_line_y'], 'top_line is true but no y-values'
            mode['top_line_y'] = list(map(lambda x: scale_factor*x, mode['top_line_y']))
            
        if mode['bot_line']:
            assert mode['bot_line_y'], 'bot_line is true but no y-values'
            mode['bot_line_y'] = list(map(lambda x: scale_factor*x, mode['bot_line_y']))

        # output the metadata in csv format
        with open(os.path.join(output_dir, image_basename+'.csv'), 'w') as f:
            f.write(StitchedImage.OUTPUT_CSV_FORMAT.format(**mode))
        
    def __init__(self, pool, image_path, scale_factor, output_dir):

        self.pool = pool
        
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.output_dir = output_dir
        
        self.image = None
        self.image_future = None
        self.image_semaphore = mp.Semaphore()

        self.mode = StitchedImage.get_base_mode()

        self.thread = None
        
    def get_mode(self):

        return self.mode

    def set_mode(self,
                 flip_x=None,
                 flip_y=None,
                 top_line=None,
                 top_line_y=None,
                 top_num_groups=None,
                 bot_line=None,
                 bot_line_y=None,
                 bot_num_groups=None,
      ):

        with self.image_semaphore: 
            if self.image is None:
                self.image = StitchedImage._load_image(self.image_path, self.scale_factor)

            if flip_x is not None:
                if self.mode['flip_x'] != flip_x:
                    self.mode['flip_x'] = flip_x
                    self.image['image_uint8'] = self.image['image_uint8'].transpose(Image.FLIP_LEFT_RIGHT)

            if flip_y is not None:
                if self.mode['flip_y'] != flip_y:
                    self.mode['flip_y'] = flip_y
                    self.image['image_uint8'] = self.image['image_uint8'].transpose(Image.FLIP_TOP_BOTTOM)

            if top_line is not None:
                self.mode['top_line'] = top_line

            if top_line_y is not None:
                self.mode['top_line_y'] = top_line_y
                
            if top_num_groups is not None:
                self.mode['top_num_groups'] = top_num_groups
                
            if bot_line is not None:
                self.mode['bot_line'] = bot_line

            if bot_line_y is not None:
                self.mode['bot_line_y'] = bot_line_y
                
            if bot_num_groups is not None:
                self.mode['bot_num_groups'] = bot_num_groups
                
    def commit_to_disk(self):

        image = self.get_image()
        image = image['image_original']

        image_basename = os.path.basename(self.image_path)

        future = self.pool.apply_async(StitchedImage._commit_to_disk, args=(image, image_basename, self.output_dir, self.scale_factor, self.mode))
        return future

    def get_image(self):

        # ensure there isn't a prefetch worker still loading the image and ensure function call is thread safe
        with self.image_semaphore: 
        
            # load if image isn't in memory yet
            if self.image is None:
                self.image = StitchedImage._load_image(self.image_path, self.scale_factor)
                    
            # perform processing
            image = self.image
            
            return image

    def evict_image(self):

        with self.image_semaphore: 
            self.image = None
            assert self.image is None
            
    def prefetch_image(self):

        if self.image is not None:
            return

        self.image_semaphore.acquire()
        self.image_future = self.pool.apply_async(StitchedImage._load_image, args=(self.image_path, self.scale_factor), callback=self._finalize_image_prefetch)

    def _finalize_image_prefetch(self, future):

        self.image = future
        #print('finished prefetching! {}'.format(self.image_path))
        self.image_semaphore.release()
        
if __name__ == '__main__':
    # some basic testing code below
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='image', type=str)
    args = parser.parse_args()

    mode = StitchedImage.get_base_mode()
    print(json.dumps(mode, indent=4))

    pool = mp.Pool(1)
    image_handle = StitchedImage(pool, args.image, 1)

    mode['flip_x'] = True
    image_handle.set_mode(**mode)
    print(json.dumps(image_handle.get_mode(), indent=4))

    # image = image_handle.get_image()
    # print(image)

    image_handle.prefetch_image()
    image = image_handle.get_image()
    print(image)
    image = image_handle.get_image()
    print(image)
    
    image_handle.evict_image()
    image_handle.prefetch_image()
    image = image_handle.get_image()
    print(image)
    image = image_handle.get_image()
    print(image)

    image = image_handle.get_image()
    print(image)
    image_handle.evict_image()

    image_handle.commit_to_disk()
    image_handle.evict_image()
    image_handle.evict_image()
    image_handle.evict_image()

    
