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
    <img src="https://user-images.githubusercontent.com/32263112/220398856-39348d77-cbd5-40bb-b22c-92d17d3e9ab0.PNG"> </a> 
</div>

## What is RealScaler?
RealScaler is a Windows app that uses Real-ESRGAN artificial intelligence to enhance, enlarge and reduce noise in photographs and videos.

## Other AI projects.🤓

https://github.com/Djdefrag/QualityScaler / QualityScaler - image/video AI upscaler app (BSRGAN)

https://github.com/Djdefrag/FluidFrames.RIFE / FluidFrames.RIFE - video AI interpolation app (RIFE-HDv3)


## Credits.

Real-ESRGAN - https://github.com/xinntao/Real-ESRGAN

## How is made. 🛠

RealScaler is completely written in Python, from backend to frontend. 
External packages are:
- AI  -> torch / torch-directml
- GUI -> tkinter / tkdnd / sv_ttk
- Image/video -> openCV / moviepy
- Packaging   -> pyinstaller
- Miscellaneous -> pywin32 / win32mica

## Requirements. 🤓
- Windows 11 / Windows 10
- RAM >= 8Gb
- Directx12 compatible GPU:
    - any AMD >= Radeon HD 7000 series
    - any Intel HD Integrated >= 4th-gen core
    - any NVIDIA >=  GTX 600 series

## Features.

- [x] Easy to use GUI
- [x] Image/list of images upscale
- [x] Video upscale
- [x] Drag&drop files [image/multiple images/video]
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Resize image/video before upscaling
- [x] Multiple gpu backend
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫

- [ ] Update 2.0 (now under development)
    - [ ] Python 3.11 (expecting ~30% more performance)
    - [ ] torch/torch-directml 2.0 (more performance)
    - [ ] a new completely redesigned graphical interface, with many more options for the user
    - [ ] upscaling of images and videos at once (currently it is possible to upscale a single image, a list of images or a single video)
    - [ ] upscale multiple videos at once
- [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
- [x] New GUI with Windows 11 style
- [x] Include audio for upscaled video
- [x] Optimizing video frame resize and extraction speed
- [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
- [x] Python 3.10 (expecting ~10% more performance)

## Known bugs.
- [x] Windows10 - the app starts with white colored navbar instead of dark
- [x] Upscaling multiple images doesn't free GPU Vram, so the it is very likely that the process will fail when the gpu memory fills up
- [x] Filenames with non-latin symbols (for example kangy, cyrillic etc.) not supported - [Temp solution] rename files like "image" or "video"
- [ ] When running RealScaler as Administrator, drag&drop is not working

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

