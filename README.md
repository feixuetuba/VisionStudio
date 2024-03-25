#Vision studio  
图像处理相关工具集合

## 依赖
- python >=3.9
- pyside6
- opencv-python 
- imageio

## 常见问题  
1. Ubuntu 上报错：Could not load the Qt platform plugin "xcb" ......Abort  
sudo apt install libxcb-cursor0
2. ubuntu 上报错：GLIBC2.8找不到
```bash
echo 'deb http://security.debian.org/debian-security buster/updates main' >> /etc/apt/sources.list
apt-get update
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 112695A0E562B32A 54404762BBB6E853
apt-get update
apt list --upgradable
apt-get install libc6-dev -y
apt-get install libc6 -y
strings /lib/x86_64-linux-gnu/libc.so.6 |grep GLIBC_
```
3. Ubuntu命令行启动上报错：qt.qpa.gl: QXcbConnection: Failed to initialize GLX
export QT_XCB_GL_INTEGRATION=none