<div align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/32263112/202846672-027bc15c-8db1-424c-b241-5b466e66c66e.png" width="175"> </a> 
    <br><br> ReSR.Scaler - image/video deeplearning upscaling for any GPU <br><br>
    <a href="https://jangystudio.itch.io/resr.scaler">
         <img src="https://user-images.githubusercontent.com/86362423/162710522-c40c4f39-a6b9-48bc-84bc-1c6b78319f01.png" width="200">
    </a>
</div>
<br>
<div align="center">
    <img src="https://user-images.githubusercontent.com/32263112/204875281-7e7945b0-321c-4d12-b293-45542c73857a.PNG"> </a> 
</div>


## Other AI projects.🤓

https://github.com/Djdefrag/QualityScaler / QualityScaler - image/video deeplearning upscaler with any GPU (BSRGAN/Pytorch)


## No Watermarks ✨
To avoid watermarks leave a tip here: https://jangystudio.itch.io/resr.scaler ❤

## Credits.

Real-ESRGAN - https://github.com/xinntao/Real-ESRGAN

## How is made. 🛠

ReSR.Scaler is completely written in Python, from backend to frontend. External packages are:
- [ ] AI  -> Pytorch-directml
- [ ] GUI -> Tkinter / Tkdnd / Sv_ttk
- [ ] Image/video -> OpenCV / Moviepy
- [ ] Packaging   -> Pyinstaller
- [ ] Miscellaneous -> Pywin32 / Win32mica / split_image

## Installation. 👨‍💻
#### Prerequisites: 
 Visual C++: https://www.techpowerup.com/download/visual-c-redistributable-runtime-package-all-in-one/
 
 DirectX runtime: https://www.microsoft.com/en-us/download/details.aspx?id=8109
 
#### Installation:
 1. download the ReSR.Scaler release .zip
 2. unzip using 7zip or similar
 3. execute ReSR.Scaler.exe in the directory

## Requirements. 🤓
- [ ] Windows 11 / Windows 10
- [ ] RAM >= 8Gb
- [ ] Directx12 compatible GPU:
    - [ ] any AMD >= Radeon HD 7000 series
    - [ ] any Intel HD Integrated >= 4th-gen core
    - [ ] any NVIDIA >=  GTX 600 series
- [ ] CPU [works without GPU, but is very slow]

## Features.

- [x] Easy to use GUI
- [x] Image/list of images upscale
- [x] Video upscale
- [x] Drag&drop files [image/multiple images/video]
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Resize image/video before upscaling
- [x] Cpu and Gpu backend
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫

- [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
- [x] New GUI with Windows 11 style
- [x] Include audio for upscaled video
- [ ] Optimizing image/frame resize and video frames extraction speed
- [ ] Update libraries 
    - [ ] Python 3.10 (expecting ~10% more performance) 
    - [ ] Python 3.11 (expecting ~30% more performance)

## Known bugs.
- [x] Windows10 - the app starts with white colored navbar instead of dark
- [x] Upscaling multiple images doesn't free GPU Vram, so the it is very likely that the process will fail when the gpu memory fills up
- [ ] Filenames with non-latin symbols (for example kangy, cyrillic etc.) not supported - [Temp solution] rename files like "image" or "video"
- [ ] When running ReSR.Scaler as Administrator, drag&drop is not working

### Examples.
![teaser](https://user-images.githubusercontent.com/32263112/202862469-ef70b5cc-3a23-496d-b4ae-59eb7fbd5a32.jpg)

![test](https://user-images.githubusercontent.com/32263112/203076458-71bf97c0-8d40-462c-b56c-7106b911e3ef.png)

![test2](https://user-images.githubusercontent.com/32263112/203076479-d382e98d-4425-4b69-9959-3e8c61baa354.png)

![test3](https://user-images.githubusercontent.com/32263112/203334168-c4c9411e-b8b0-4ba2-aa7b-9f2c2c5d4be8.png)

![test4](https://user-images.githubusercontent.com/32263112/203338120-fd4c1ddd-b4ba-4ad2-b8fa-0d8cd92689f0.png)

![test5](https://user-images.githubusercontent.com/32263112/203338133-0d0945f1-0129-4b36-8801-1510cf8892b8.png)


