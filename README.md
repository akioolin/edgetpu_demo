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

這是以EdgeTPU C++ API為基礎的C++範例。分為二個部份，一個是將edgetpu c++ api編譯成為靜態函式庫。另外一個是範例程式，只有使用edgetpu c++ api, tf-lite api以及其相關函式庫，再來就是OpenCV。

要執行這個範例，需要把EdgeTPU Dev-Board更新到Mendel 4.0。相關更新步驟，請參考edgetpu網站的文件步驟進行。
更新好之後，請用以下的命令來安裝相關的套件：

sudo apt install mc joe samba libhdf5-103 libqtgui4 libatlas-base-dev libqt4-test python3-opencv libopencv-dev astyle libncurses-dev ntpdate git cmake
