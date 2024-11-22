<div align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/32263112/202846672-027bc15c-8db1-424c-b241-5b466e66c66e.png" width="175"> </a> 
    <br><br> RealScaler - image/video AI upscaler app (Real-ESRGAN) <br><br>
    <a href="https://jangystudio.itch.io/realesrscaler">
        <button>
            <img src="https://static.itch.io/images/badge-color.svg" width="225" height="70">
        </button>     
    </a>
</div>
<br>
<div align="center">
    <img src="https://github.com/user-attachments/assets/a9c8e060-c20d-4c57-978d-5cd92b988f62"> </a> 
</div>

## What is RealScaler?
RealScaler is a Windows app powered by RealESRGAN AI to enhance, upscale and de-noise photos and videos.

## Other AI projects.🤓
- https://github.com/Djdefrag/QualityScaler / QualityScaler - image/video AI upscaler app
- https://github.com/Djdefrag/FluidFrames.RIFE / FluidFrames.RIFE - video AI interpolation app (RIFE-HDv3)

## Credits.
Real-ESRGAN - https://github.com/xinntao/Real-ESRGAN

## How is made. 🛠
RealScaler is completely written in Python, from backend to frontend. 
- [x] pytorch (https://github.com/pytorch/pytorch)
- [x] onnx (https://github.com/onnx/onnx)
- [x] onnxconverter-common (https://github.com/microsoft/onnxconverter-common)
- [x] onnxruntime-directml (https://github.com/microsoft/onnxruntime)
- [x] customtkinter (https://github.com/TomSchimansky/CustomTkinter)
- [x] openCV (https://github.com/opencv/opencv)
- [x] moviepy (https://github.com/Zulko/moviepy)
- [x] pyInstaller (https://github.com/pyinstaller/pyinstaller)

## Make it work by yourself. 👨‍💻
Prerequisites.
- Python installed on your pc (https://www.python.org/downloads/release/python-3119/)
- VSCode installed on your pc (https://code.visualstudio.com/)
- AI models downloaded (https://gofile.io/d/yaMlZO)
- FFMPEG.exe downloaded (https://www.gyan.dev/ffmpeg/builds/) RELEASE BUILD > ffmpeg-release-essentials.7z

Getting started.
- Download the project on your PC (Green button Code > Download ZIP)
- Extract the project from the .zip
- Extract the AI models files in /AI-onnx folder
- Extract FFMPEG.exe in /Assets folder
- Open the project with VSCode (Drag&Drop the project directory on VSCode)
- Click RealScaler.py from left bar (VSCode will ask to install Python plugins)
- Install dependencies. In VSCode there is the "Terminal" panel, click there and execute the command "pip install -r requirements.txt"
- Close VSCode and re-open it (this will refresh all the dependecies installed)
- Click on the "Play button" in the upper right corner of VSCode

## Requirements. 🤓
- Windows 11 / Windows 10
- RAM >= 8Gb
- Any Directx12 compatible GPU with >= 4GB VRAM

## Features.
- [x] Elegant and easy to use GUI
- [x] Image and Video upscale
- [x] Multiple GPUs support
- [x] Compatible images - jpg, png, tif, bmp, webp, heic
- [x] Compatible video - mp4, wemb, mkv, flv, gif, avi, mov, mpg, qt, 3gp
- [x] Automatic image tilling to avoid gpu VRAM limitation
- [x] Resize image/video before upscaling
- [x] Interpolation beetween original file and upscaled file
- [x] Video upscaling STOP&RESUME
- [x] PRIVACY FOCUSED - no internet connection required / everything is on your PC

## Next steps. 🤫
- [x] 1.X versions
    - [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
    - [x] New GUI with Windows 11 style
    - [x] Include audio for upscaled video
    - [x] Optimizing video frame resize and extraction speed
    - [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
    - [x] Python 3.10 (expecting ~10% more performance)
- [x] 2.X versions
    - [x] New, completely redesigned graphical interface based on @customtkinter
    - [x] Upscaling images and videos at once (currently it is possible to upscale images or single video)
    - [x] Upscale multiple videos at once
    - [x] Choose upscaled video extension
    - [x] Interpolation between the original and upscaled image/video
    - [x] More Interpolation levels (Low, Medium, High)
    - [x] Show the remaining time to complete video upscaling
    - [x] Support for SRVGGNetCompact AI architecture
    - [x] Metadata extraction and application from original file to upscaled file (via exiftool)
- [ ] 3.X versions
    - [x] New AI engine powered by onnxruntime-directml (https://github.com/microsoft/onnxruntime))
    - [x] Python 3.11 (performance improvements)
    - [x] Python 3.12 (performance improvements)
    - [x] Display images/videos upscaled resolution in the GUI
    - [x] Updated FFMPEG to version 7.x (latest release)
    - [x] Saving user settings (AI model, GPU, CPU  etc.)
    - [x] Video multi-threading AI upscale 
    - [x] Video upscaling STOP&RESUME

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

