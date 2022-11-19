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
    <img src="https://user-images.githubusercontent.com/32263112/202846779-fb3e26a3-de58-4266-a21f-5697236bc96c.PNG"> </a> 
</div>

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

## My testing PC.
- [ ] Windows 10 ReviOS
- [ ] CPU Ryzen 5600G
- [ ] RAM 16Gb
- [ ] GPU Nvidia 1660
- [ ] STORAGE 1 Sata 120Gb SSD, 1 NVME 500Gb SSD

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
- [ ] When running QualityScaler as Administrator, drag&drop is not working

