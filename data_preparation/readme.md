# Dataset Preparation

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

* Specify the file path to [caricature.txt](./caricature.txt) in [Line 6 of align_caricature_data.py](./align_caricature_data.py#L6).
* Specify the folder path to the FacialPoints of WebCaricature in [Line 7 of align_caricature_data.py](./align_caricature_data.py#L7).
* Specify the folder path to the OriginalImages of WebCaricature in [Line 8 of align_caricature_data.py](./align_caricature_data.py#L8).
* Specify the folder path to save the aligned images in [Line 9 of align_caricature_data.py](./align_caricature_data.py#L9).
* Run the script [align_caricature_data.py](./align_caricature_data.py) to produce 199 aligned 256\*256 caricature face images. 
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

<br/>

## Anime Dataset

![anime_overview](https://user-images.githubusercontent.com/18130694/158095492-e5533fe2-586a-419b-a03d-bee6970a243f.jpg)

### Download
 
Please download the raw images from [Danbooru Portraits](https://www.gwern.net/Crops#danbooru2019-portraits).

### Face Alignment

* Specify the file path to [anime.txt](./anime.txt) in [Line 6 of align_anime_data.py](./align_anime_data.py#L6).
* Specify the folder path to the Danbooru Portrait Dataset in [Line 7 of align_anime_data.py](./align_anime_data.py#L7).
* Specify the folder path to save the aligned images in [Line 8 of align_anime_data.py](./align_anime_data.py#L8).
* Run the script [align_anime_data.py](./align_anime_data.py) to produces 174 aligned 512\*512 anime face images. 
```python
python align_anime_data.py
```

### Face Super-Resolution

* Upsample the aligned images to 1024\*1024 by applying [waifu2x](https://github.com/YukihoAA/waifu2x_snowshell/releases).
* We use waifu2x-converter-cpp.exe and the super resolution parameters:
```python
.\waifu2x-converter-cpp --noise-level 1 -i Path_To_512_Images --scale-ratio 2 -r 1 -o Path_To_1024_Images -g 1 -a 0
```

<br/>

## Build Your Own Dataset

<!--![arcane-overview](https://user-images.githubusercontent.com/18130694/158124926-2e53861d-3814-485d-ad9f-d45a339dd7fe.jpg)-->


We use face detection and face alignment to automatically collect artistic face images from cartoon moives like [Arcane](https://www.netflix.com/sg/title/81435684?source=35).

Suitable for artistic portraits that look like real human faces.

### Download
 
Please download the source moive.

Please download the face detection model:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```
### Face Detection and Alignment

* Specify the file name of the images to be saved in [Line 15 of find_face_in_video.py](./find_face_in_video.py#L15).
* Specify the file path to the moive in [Line 16 of find_face_in_video.py](./find_face_in_video.py#L16).
* Specify the folder path to save the found face images in [Line 17 of find_face_in_video.py](./find_face_in_video.py#L17).
* Specify the model path to the downloaded shape_predictor_68_face_landmarks.dat in [Line 18 of find_face_in_video.py](./find_face_in_video.py#L18).
* Specify the length of the black letterboxing of the moive in [Line 19 of find_face_in_video.py](./find_face_in_video.py#L19).
* Run the script [find_face_in_video.py](./find_face_in_video.py) to find and crop aligned 512\*512 face images. 
```python
python find_face_in_video.py
```

### Face Super-Resolution

* Upsample the images to 1024\*1024 by applying [waifu2x](https://github.com/YukihoAA/waifu2x_snowshell/releases).
* We use waifu2x-converter-cpp.exe and the super resolution parameters:
```python
.\waifu2x-converter-cpp --noise-level 1 -i Path_To_512_Images --scale-ratio 2 -r 1 -o Path_To_1024_Images -g 1 -a 0
```

### Optional Post-Processing

* Mannually filter wrong detections and low-quality ones.
* Use PhotoShop to [remove motion blurs](https://helpx.adobe.com/sg/photoshop/using/reduce-camera-shake-induced-blurring.html) in movies.
* Adjust color and tone of the dark images.

### Useful Resources

* [arcane.txt](./arcane.txt): filenames of our filtered 100 arcane face images.
  - E.g., the filename `1_016_04488.jpg` means the 16th detected faces, the 4488th frames of the Arcane Episode 1.
* [pixar.txt](./pixar.txt): YouTube video links where we collected Pixar face images.
