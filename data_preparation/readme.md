# Caricature and Anime Dataset Preparation

## Caricature Dataset

![caricature_overview](https://user-images.githubusercontent.com/18130694/158067472-812df136-a4d2-485b-985c-09be27608fe3.jpg)

### Download
 
Please download the raw data from [WebCaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm).

Unzip the downloaded **WebCaricature.zip**, which is in the following folder structure:
```
WebCaricature
|--EvaluationProtocols
|--FacialPoints
   <landmarks of the caricatures>
|--Filenames
|--OriginalImages
   <raw photo and caricature images>
|--Readme.txt
```

### Face Alignment

* Specify the file path to the [caricature.txt](./caricature.txt) in Line 6 of [align_caricature_data.py](./align_caricature_data.py).
* Specify the folder path to the FacialPoints of WebCaricature in Line 7 of [align_caricature_data.py](./align_caricature_data.py).
* Specify the folder path to the OriginalImages of WebCaricature in Line 8 of [align_caricature_data.py](./align_caricature_data.py).
* Specify the folder path to save the aligned images in Line 9 of [align_caricature_data.py](./align_caricature_data.py).
* Run the script [align_caricature_data.py](./align_caricature_data.py) to produces 199 aligned 256\*256 caricature face images. 
```python
python align_caricature_data.py
```

### Face Super-Resolution

* Upsample the aligned images to 1024\*1024 by applying [waifu2x](https://github.com/YukihoAA/waifu2x_snowshell/releases) twice.
* We use waifu2x-converter-cpp.exe and the super resolution parameters:
```python
.\waifu2x-converter-cpp --noise-level 1 -i Path_To_256_Images --scale-ratio 2 -r 1 -o Path_To_512_Images -g 1 -a 0
.\waifu2x-converter-cpp --noise-level 1 -i Path_To_512_Images --scale-ratio 2 -r 1 -o Path_To_1024_Images -g 1 -a 0
```
