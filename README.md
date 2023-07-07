<div align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/32263112/202846672-027bc15c-8db1-424c-b241-5b466e66c66e.png" width="175"> </a> 
    <br><br> RealScaler - fast image/video AI upscaler app (Real-ESRGAN) <br><br>
    <a href="https://jangystudio.itch.io/realesrscaler">
         <img src="https://user-images.githubusercontent.com/86362423/162710522-c40c4f39-a6b9-48bc-84bc-1c6b78319f01.png" width="200">
    </a>
</div>
<br>
<div align="center">
    <img src="https://github.com/Djdefrag/RealScaler/assets/32263112/4e1c0a8a-b02c-4736-9ce1-c548b47f90c2"> </a> 
</div>


## What is RealScaler?
RealScaler is a Windows app that uses Real-ESRGAN artificial intelligence to enhance, enlarge and reduce noise in photographs and videos.

## Other AI projects.🤓

- https://github.com/Djdefrag/QualityScaler / QualityScaler - image/video AI upscaler app (BSRGAN)
- https://github.com/Djdefrag/FluidFrames.RIFE / FluidFrames.RIFE - video AI interpolation app (RIFE-HDv3)


## Credits.

Real-ESRGAN - https://github.com/xinntao/Real-ESRGAN

## How is made. 🛠

RealScaler is completely written in Python, from backend to frontend. 
External packages are:
- AI  -> torch / torch-directml
- GUI -> customtkinter / win32mica
- Image/video -> openCV / moviepy
- Packaging   -> pyinstaller / upx

## Requirements. 🤓
- Windows 11 / Windows 10
- RAM >= 8Gb
- Any Directx12 compatible GPU with >= 4GB VRAM

## Features.
- [x] Easy to use GUI
- [x] Images and Videos upscale
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Resize image/video before upscaling
- [x] Multiple Gpu support
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫
- [ ] 1.X versions
    - [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
    - [x] New GUI with Windows 11 style
    - [x] Include audio for upscaled video
    - [x] Optimizing video frame resize and extraction speed
    - [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
    - [x] Python 3.10 (expecting ~10% more performance)
- [ ] 2.X versions (now under development)
    - [x] New, completely redesigned graphical interface based on @customtkinter
    - [x] Upscaling images and videos at once (currently it is possible to upscale images or single video)
    - [x] Upscale multiple videos at once
    - [ ] Python 3.11 (expecting ~30% more performance)
    - [ ] Torch/torch-directml 2.0 (expecting ~20% more performance)

### Some Examples.
#### Videos
- Original

![boku-no-hero-my-hero-academia](https://user-images.githubusercontent.com/32263112/209639439-94c8774d-354e-4d56-9123-e1aa4af95e08.gif)

- RealESRGANx4 - upscaled in 3 minutes and 23 seconds

https://user-images.githubusercontent.com/32263112/209639499-83eb4609-a842-43f9-b8a2-9fffd23e1d2c.mp4

- RealESR_Gx4 - upscaled in 57 seconds

https://user-images.githubusercontent.com/32263112/209639569-c201a965-c6bf-4b7c-9904-61114b5bf4d5.mp4


#### Images!

![test1](https://user-images.githubusercontent.com/32263112/223775329-2400f251-d6a3-45bb-ae94-09e40c55a6e1.png)

![test2](https://user-images.githubusercontent.com/32263112/223775065-2c304b76-ca1b-4efc-83d5-16c091be0cd1.png)

![test5](https://user-images.githubusercontent.com/32263112/203338133-0d0945f1-0129-4b36-8801-1510cf8892b8.png)

![a](https://user-images.githubusercontent.com/32263112/206723952-3f3110c9-9328-4bcc-94e0-8aaec0279eeb.png)

