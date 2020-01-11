# edgetpu_demo
C++ based EdgeTPU C++ API Demo code

This project contains two part.

The first part is EdgeTPU C++ API building project. about tensorflow-lite dependency, 
please refer to https://github.com/google-coral/edgetpu/blob/master/WORKSPACE#L5. 
other dependency libraries header file are copied from the respect project or from 
tensorflow-lite downloads.

The second part is based on first part's outcome of EdgeTPU C++ API project. this 
project is a pure c++ demo application which shows how to use the EdgeTPU C++ API 
with OpenCV, which is run on EdgeTPU Dev-board with Mendel 4.0 distribution.
