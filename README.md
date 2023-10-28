# RunJetsonNano4.5.1
A guide how to run simple linear layer and CNN on Jetson Nano with JetPack

##Update
1. Get the container from jetson-voice
```
git clone --branch dev https://github.com/dusty-nv/jetson-voice
cd jetson-voice
docker/run.sh
```
2. Get the repo from Eric
```
git clone https://github.com/eric0708/ODML_Lab2/tree/main
```
3. Run the inference


Follow [NV's guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup) to connect to your Jetson. In MacOS, it's
```
$ ls /dev/cu.usbmodem*
/dev/cu.usbmodem14133200001053
$ sudo screen /dev/cu.usbmodem14133200001053 115200
```

First, download the image from https://developer.nvidia.com/embedded/jetpack-sdk-451-archive
- choose `JETSON NANO DEVELOPER KITS` - `For Jetson Nano Developer Kit`
Follow https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write to instore the image
Follow https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch to use container
```
sudo docker pull nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
sudo docker run -it --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
```
Now you are in the container(ubuntu OS)
install vim
```
apt-get update
apt-get install vim
```
Pull this repo
```
git clone https://github.com/weilun-chiu/RunJetsonNano4.5.1
```
Run test
```
python3 check_version.py
python3 linear_example.py
python3 simpleCNN.py
```
Use the following wheel to get scipy (from https://github.com/jetson-nano-wheels/python3.6-scipy-1.5.4)
```
pip3 install 'https://github.com/jetson-nano-wheels/python3.6-scipy-1.5.4/releases/download/v0.0.1/scipy-1.5.4-cp36-cp36m-linux_aarch64.whl'
```
