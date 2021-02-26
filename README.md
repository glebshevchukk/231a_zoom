Pipeline

1. Capture photos using iPhone portrait mode
2. Transfer photos to img/ folder
3. Install ExifTool at https://exiftool.org/
4. Use it to get the depth information from images:
```
exiftool -b -MPImage2 0.jpg > 0_depth.jpg 
```
