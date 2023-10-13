# RunJetsonNano4.5.1
A guide how to run simple linear layer and CNN on Jetson Nano with JetPack

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
