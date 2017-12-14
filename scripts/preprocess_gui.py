#!/usr/bin/env python3

import argparse
import numpy as np
import threading
import time

import tkinter as tk
from PIL import Image, ImageTk

class GUI:


    def __init__(self):
        self._root = tk.Tk()
        self._canvas = tk.Canvas(self._root, width=0, height=0)

        self._counter = 0
        self._photoImg = None

        # set callbacks
        self._root.bind("<Key>", lambda x: self._key(x))
        self._canvas.bind("<Button-1>", lambda x: self._click(x))

        # kick off line drawing thread
        t = threading.Thread(target=self._loop)
        t.start()

        # img = Image.open("/home/jemmons/test.png")
        # self._photoImg = ImageTk.PhotoImage(img)
        # self._canvas.create_image(0, 0, image=self._photoImg, anchor=tk.NW)

        self._canvas.pack(fill=tk.BOTH, expand=tk.YES) 
        self._clear()
        
        self._root.mainloop()
        
    def _click(self, event):

        print("clicked at", event.x, event.y)
        line = self._canvas.create_line(0, event.y, 1000, event.y)
        self._counter += 1

        print(self._counter)
        if(self._counter % 4 == 0 and self._counter != 0):
            time.sleep(0.25)
            self._clear()

    def _key(self, event):
        print("pressed", repr(event.char))

    def _loop(self):
        while True:
            x, y = self._get_cursor_pos()
            line = self._canvas.create_line(0, y, 1000, y)
            time.sleep(0.05)
            self._canvas.delete(line)

    def _get_cursor_pos(self):
        x = self._root.winfo_pointerx()
        y = self._root.winfo_pointery()
        abs_coord_x = self._root.winfo_pointerx() - self._root.winfo_rootx()
        abs_coord_y = self._root.winfo_pointery() - self._root.winfo_rooty()

        return abs_coord_x, abs_coord_y

    def _clear(self):
        self._canvas.delete('all')

        img = Image.open("/home/jemmons/test.png")
        self._photoImg = ImageTk.PhotoImage(img)
        self._canvas.create_image(0, 0, image=self._photoImg, anchor=tk.NW)
        w, h = img.size
        self._canvas.config(width=w, height=h)
        
        
def main(args):
    GUI()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing GUI to make analysis easier.')

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
