## Face key point detection and synthetic mask generation



- Use `dlib` to realize face key points detection.
- Add synthetic masks to the face.



## Requirements:

- python 3.6: you'd better create a new conda environment with python == 3.6  

- opencv: `pip install opencv-python`

- tqdm: `pip install tqdm`

- dlib: `pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl`

  note: `dlib-19.8.1-cp36-cp36m-win_amd64.whl` has been given in this repository



## Start:

Input an image and output the image with mask:

```
python main.py --mode mask --image data/test.jpg
```

Input the folder path of images in order to add masks to all the images:

```
python main.py --mode mask_folder  --input_folder ... --output_folder ...
```

Input an image and output the image with its key points and RoI:

```
python main.py --mode keypoint -- image data/test.jpg
```





## Result:

##### add mask:



<center class="half">
    <img src="figure\test.jpg" alt="test" style="zoom:50%;" />
    <img src="figure\test.jpg_with_mask.png" alt="test.jpg_with_mask" style="zoom:50%;" />
</center>



##### key point:

   <center class="half">
       <img src="figure\result1.png" alt="result1" style="zoom:50%;" />
       <img src="figure\result2.png" alt="result2" style="zoom:50%;" />
       <img src="figure\result3.png" alt="result3" style="zoom:50%;" />
       <img src="figure\result4.png" alt="result4" style="zoom:50%;" />
       <img src="figure\result5.png" alt="result5" style="zoom:50%;" />
       <img src="figure\result6.png" alt="result6" style="zoom:50%;" />
       <img src="figure\result7.png" alt="result7" style="zoom:50%;" />
       <img src="figure\result8.png" alt="result8" style="zoom:50%;" />



​       


  

   

​       















