#!/usr/bin/env python3

import argparse
import cv2
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


    def __init__(self, images, scale_factor):

        pool = mp.Pool(48)
        self._images = list(map(lambda x: stitched_image.StitchedImage(pool, x, scale_factor), images))

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
        
        self._counter = 0
        self._photoImg = None
        self._img_num = 0
        
        # set callbacks
        self._root.bind("<Key>", lambda x: self._key(x))
        self._canvas.bind("<Button-1>", lambda x: self._click(x))

        # kick off line drawing thread 
        self._start = False
        t = threading.Thread(target=self._loop)
        t.start()

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

        for line in self._prev_lines:
            self._canvas.delete(line)

        if self._prev_lines:
            self._prev_lines = []
            self._mode['top_line_y'] = []
            self._mode['bot_line_y'] = []
        
        num_lines = 2 if self._mode['top_line'] else 0
        num_lines += 2 if self._mode['bot_line'] else 0

        if self._mode['top_line']:
            if self._counter < 2:
                self._mode['top_line_y'].append(event.y)
            elif self._mode['bot_line_y']:
                self._mode['bot_line_y'].append(event.y)
        else:
            if self._mode['bot_line'] and self._counter < 2:
                self._mode['bot_line_y'].append(event.y)

        if(num_lines == 0):
            print('no lines set on image!')
            return
            
        line = self._canvas.create_line(0, event.y, self._w, event.y, width=2)
        self._counter += 1

        if(self._counter % num_lines == 0 and self._counter != 0):
            self._log()
            self._forward()
            self._load_image()
            self._draw_prev_lines()
            
    def _key(self, event):
        print("pressed", repr(event.char))

        self._counter = 0

        if event.char == '\r':
            self._log()
            self._load_image()
            self._draw_prev_lines()
            return 
            
        elif event.char == '6':
            self._log()
            self._forward()
            self._load_image()
            self._draw_prev_lines()
            return 
            
        elif event.char == '4':
            self._log()
            self._backward()
            self._load_image()
            self._draw_prev_lines()
            return 

        elif event.char == 't':
            self._mode['top_line'] = not self._mode['top_line']
            if not self._mode['top_line']:
                self._mode['top_line_y'] = []
            self._log()
            self._load_image()
            return 
        
        elif event.char == 'b':
            self._mode['bot_line'] = not self._mode['bot_line']
            if not self._mode['bot_line']:
                self._mode['bot_line_y'] = []
            self._log()
            self._load_image()
            return 
        
        elif event.char == 'x':
            self._mode['flip_x'] = not self._mode['flip_x']
            self._log()
            img = self._images[self._img_idx].set_mode(**self._mode)

            self._load_image()
            return 
            
        elif event.char == 'y':
            self._mode['flip_y'] = not self._mode['flip_y']
            self._log()
            img = self._images[self._img_idx].set_mode(**self._mode)
            self._load_image()
            return 

        print(json.dumps(self._mode, indent=4))
            
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
        self._img_idx += 1

    def _backward(self):
        if self._img_idx > 0:
            self._img_idx -= 1
    
    def _load_image(self):

        self._canvas.delete('all')

        t1 = timeit.default_timer()

        img = self._images[self._img_idx].get_image()
        self._images[self._img_idx].set_mode(**self._mode)
        img = img['image_uint8']
        # if self._img_idx > 24:
        #     self._images[self._img_idx-24].evict_image()

        t2 = timeit.default_timer()
        print('image load time: {}'.format(t2-t1))
        
        t1 = timeit.default_timer()

        self._photoImg = ImageTk.PhotoImage(img)
        self._canvas.create_image(0, 0, image=self._photoImg, anchor=tk.NW)
        self._w, self._h = img.size
        self._canvas.config(width=self._w, height=self._h)

        self._img_num += 1 
        #self._images[self._img_idx + self.prefetch_amount].prefetch_image()
        canvas_id = self._canvas.create_text((50,100), text='hello world {}'.format(self._img_num), anchor="nw")

        t2 = timeit.default_timer()
        print('page load time: {}'.format(t2-t1))
        print('READY')
        
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
        print(json.dumps(self._mode, indent=4, sort_keys=True))
        
def main(args):
    GUI(args.images, args.scale_factor)
        
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
                        help='the scaling factor to apply to the images',
                        default=1)

    parser.add_argument(dest='images',
                        metavar='image',
                        type=str,
                        nargs='+',
                        help='images to process (NOTE: processed in the order they appear)')

    args = parser.parse_args()
        
    main(args)
