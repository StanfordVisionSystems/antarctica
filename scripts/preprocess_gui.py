#!/usr/bin/env python3

import argparse
import cv2
import copy
import numpy as np
import multiprocessing as mp
import json
import threading
import time
import timeit

import stitched_image

import tkinter as tk
from PIL import Image, ImageTk

class GUI:


    def __init__(self, images, scale_factor, output_dir):

        pool = mp.Pool(48)
        self._images = list(map(lambda x: stitched_image.StitchedImage(pool, x, scale_factor, output_dir), images))
        self.output_dir = output_dir
        
        #self.prefetch_amount = 10*48
        #self._images[0].prefetch_image()
        # for i in range(1, self.prefetch_amount):
        #     self._images[i].prefetch_image()

        for image in self._images:
            image.prefetch_image()
        
        self._img_idx = 0
        
        self._root = tk.Tk()
        self._canvas = tk.Canvas(self._root, width=0, height=0)

        self._h = 0
        self._w = 0
        
        self._click_counter = 0
        self.__photoImg = None # need to keep reference to image externally because tk doesn't!
        
        # set callbacks
        self._root.bind("<Key>", lambda x: self._key(x))
        self._canvas.bind("<Button-1>", lambda x: self._click(x))

        # kick off line drawing thread 
        self._start = False
        self._t = threading.Thread(target=self._loop)
        self._t.start()

        # img = Image.open("/home/jemmons/test.png")
        # self._photoImg = ImageTk.PhotoImage(img)
        # self._canvas.create_image(0, 0, image=self._photoImg, anchor=tk.NW)

        self._canvas.pack(fill=tk.BOTH, expand=tk.YES) 

        self._prev_lines = []
        self._line_vals = []
        self._mode = {
            'flip_x' : False,
            'flip_y' : False,
            'top_line' : False,
            'top_line_y' : [],
            'top_num_groups' : 0, 
            'bot_line' : False, 
            'bot_line_y' : [],
            'bot_num_groups' : 0,
        }
        self._load_image()

        self._start = True
        self._root.mainloop()
        
    def _click(self, event):
        print(event.y)

        # delete all lines from the image on click
        num_lines = 2 if self._mode['top_line'] else 0
        num_lines += 2 if self._mode['bot_line'] else 0

        if num_lines == 0 or self._click_counter % num_lines == 0:
            for line in self._prev_lines:
                self._canvas.delete(line)

            self._prev_lines = []
            self._mode['top_line_y'] = []
            self._mode['bot_line_y'] = []
            self._click_counter = 0
            
        if(num_lines == 0):
            print('no lines set on image!')
            return

        # update the mode according to the clicks
        if self._mode['top_line']:
            if self._click_counter == 0 or self._click_counter == 1:
                self._mode['top_line_y'].append(event.y)

            elif self._click_counter == 2 or self._click_counter == 3:
                assert self._mode['bot_line']
                self._mode['bot_line_y'].append(event.y)

        elif self._mode['bot_line']:
            if self._click_counter == 0 or self._click_counter == 1:
                self._mode['bot_line_y'].append(event.y)            

        line = self._canvas.create_line(0, event.y, self._w, event.y, width=2)
        self._click_counter += 1

        if self._click_counter == num_lines:
            self._images[self._img_idx].set_mode(**self._mode)
            self._log()
            
    def _key(self, event):
        print("pressed", repr(event.char))

        self._counter = 0

        if event.char == '\r':
            self._load_image()
            self._draw_prev_lines()
            self._log()
            
        elif event.char == '6':
            num_lines = 2 if self._mode['top_line'] else 0
            num_lines += 2 if self._mode['bot_line'] else 0
            if self._click_counter != 0 and self._click_counter < num_lines:
                print('need to finish clicking lines before you continue: {} of {}'.format(self._click_counter, num_lines))
                return
            
            self._forward()
            self._load_image()
            self._draw_prev_lines()
            self._log()
            
        elif event.char == '4':
            num_lines = 2 if self._mode['top_line'] else 0
            num_lines += 2 if self._mode['bot_line'] else 0
            if self._click_counter != 0 and self._click_counter < num_lines:
                print('need to finish clicking lines before you continue: {} of {}'.format(self._click_counter, num_lines))
                return

            self._backward()
            self._load_image()
            self._draw_prev_lines()
            self._log()

        elif event.char == 't':
            self._mode['top_line'] = not self._mode['top_line']
            if not self._mode['top_line']:
                self._mode['top_line_y'] = []
            self._images[self._img_idx].set_mode(**self._mode)
            self._click_counter = 0 

            self._load_image()
            self._log()
        
        elif event.char == 'b':
            self._mode['bot_line'] = not self._mode['bot_line']
            if not self._mode['bot_line']:
                self._mode['bot_line_y'] = []

            self._images[self._img_idx].set_mode(**self._mode)
            self._click_counter = 0 

            self._load_image()
            self._log()
            
        elif event.char == 'x':
            self._mode['flip_x'] = not self._mode['flip_x']
            self._log()
            self._images[self._img_idx].set_mode(**self._mode)
            self._click_counter = 0 

            self._load_image()
            
        elif event.char == 'y':
            self._mode['flip_y'] = not self._mode['flip_y']
            
            self._images[self._img_idx].set_mode(**self._mode)
            self._click_counter = 0 

            self._load_image()
            self._log()

        elif event.char == 'w':
            print('commit all images to disk: {}'.format(self.output_dir))
            futures = []
            for i in range(len(self._images)):
                f = self._images[i].commit_to_disk()
                futures.append(f)

            for i in range(len(futures)):
                if i % 10 == 0:
                    print('processed {} of {}'.format(i, len(self._images)))                    
                futures[i].get()
                
            print('done!')
            
        elif event.char == 'q':
            print('quitting!')
            self._root.quit()
                        
    def _loop(self):
        
        while True:
            if self._start:
                break
            time.sleep(0.1)
            
        while True:
            x, y = self._get_cursor_pos()
            line = self._canvas.create_line(0, y, self._w, y, width=2)
            time.sleep(0.05)
            self._canvas.delete(line)

    def _get_cursor_pos(self):

        x = self._root.winfo_pointerx()
        y = self._root.winfo_pointery()
        abs_coord_x = self._root.winfo_pointerx() - self._root.winfo_rootx()
        abs_coord_y = self._root.winfo_pointery() - self._root.winfo_rooty()

        return abs_coord_x, abs_coord_y

    def _forward(self):
        if self._img_idx < len(self._images)-1:
            self._img_idx += 1
            self._images[self._img_idx].set_mode(**self._mode)

        self._click_counter = 0 
            
    def _backward(self):
        if self._img_idx > 0:
            self._img_idx -= 1

        self._mode = copy.deepcopy(self._images[self._img_idx].get_mode())
        self._click_counter = 0 
            
    def _load_image(self):

        self._canvas.delete('all')

        t1 = timeit.default_timer()
        img = self._images[self._img_idx].get_image()

        img = img['image_uint8']
        # if self._img_idx > 24:
        #     self._images[self._img_idx-24].evict_image()

        t2 = timeit.default_timer()
        # print('image load time: {}'.format(t2-t1))
        
        t1 = timeit.default_timer()

        self.__photoImg = ImageTk.PhotoImage(img)
        self._canvas.create_image(0, 0, image=self.__photoImg, anchor=tk.NW)
        self._w, self._h = img.size
        self._canvas.config(width=self._w, height=self._h)

        #self._images[self._img_idx + self.prefetch_amount].prefetch_image()
        canvas_id = self._canvas.create_text((10,20), text='img_num: {} / {}'.format(self._img_idx, len(self._images)-1), anchor="nw")

        t2 = timeit.default_timer()
        # print('page load time: {}'.format(t2-t1))
        # print('READY')
        
    def _draw_prev_lines(self):
        
        if self._mode['top_line']:
            y1, y2 = self._mode['top_line_y']
            self._prev_lines.append(self._canvas.create_line(0, y1, self._w, y1, width=2))
            self._prev_lines.append(self._canvas.create_line(0, y2, self._w, y2, width=2))

        if self._mode['bot_line']:
            y1, y2 = self._mode['bot_line_y']
            self._prev_lines.append(self._canvas.create_line(0, y1, self._w, y1, width=2))
            self._prev_lines.append(self._canvas.create_line(0, y2, self._w, y2, width=2))
                
    def _log(self):
        print('\033[2J') # clear terminal char
        print('image number {} of {}'.format(self._img_idx, len(self._images)))
        print(json.dumps(self._mode, indent=4, sort_keys=True))
        
def main(args):
    GUI(args.images, args.scale_factor, args.output_dir)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing GUI to make analysis easier.')

    parser.add_argument('--output_dir',
                        dest='output_dir',
                        metavar='output_dir',
                        type=str,
                        help='the directory to output processed images and CSV results')

    parser.add_argument('--scale_factor',
                        dest='scale_factor',
                        metavar='scale_factor',
                        type=float,
                        help='the scaling factor to apply to the images (1 / scale_factor is applied to x and y axes)',
                        default=10)

    parser.add_argument(dest='images',
                        metavar='image',
                        type=str,
                        nargs='+',
                        help='images to process (NOTE: processed in the order they appear)')

    args = parser.parse_args()
        
    main(args)
