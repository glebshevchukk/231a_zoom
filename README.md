This is our 231A class project testing different approaches to computational dolly zoom.

Example usage:
'''
python main.py --extension .jpg --interpolation_type linear --num_images 5 --threshold 140 --mapping_type full_affine --algo_type affine --image_dir img/board/    
'''
'''
python main.py --extension .jpg --interpolation_type flow  --num_images 5 --threshold 140 --mapping_type full_affine --algo_type spline --image_dir img/tree/    
'''
Instructions for using Portrait mode depth:
1. Capture photos using iPhone portrait mode
2. Transfer photos to img/ folder
3. Install ExifTool at https://exiftool.org/
4. Use it to get the depth information from images:
```
exiftool -b -MPImage2 0.jpg > 0_depth.jpg 
```

The main runner (main.py) has a number of flags:

image_dir - directory that we pull images from, expects images to be labelled as 0.jpg, 1.jpg, etc.
num_images - the number of images you want to use from that directory. If this number is larger than the amount present, the program will exit with an error
save_name - the name under which we will save the final dollied GIF in the image directory
extension - the extension of images we want to use, .jpg or .png
height - height of each image (preconfigured to iPhone image size)
width - width of each image (preconfigured to iPhone image size)

has_depth - if this is True, then we expect there to also be 0_depth.jpg, 1_depth.jpg, etc. in the image directory, created using the above exiftool process
threshold - if has_depth is True, then we threshold all images by depth that have pixel-values greater than this threshold (so we expect it to be a number between 0 and 255)
pre_segmented - if this is True, then use pre-segmented images made using the Watershed algorithm explained above.
save_orig_name - name of the original image sequence saved as a GIF in the image_dir
manual_segmentation - if this is True, then we perform segmentation using Watershed directly from this runner (expecting user input)

algo_type: affine/homography/spline, specifies the transform method we use
interpolation_type: linear/flow, specifies the inteprolation method we use at the end

debug: if True, produces plots of intermediate results like feature matching and flow, depending on what algo_type and interpolation_type are specified
mapping_type: this is an experimental flag that was used to see if seperately processing the background and foreground produced better results than processing them together, this can be ignored.
