# Create dataset from iPhone video:

> mkdir -p datasets/<NAME>/input  

> ffmpeg -i datasets/<NAME>.MOV        -vf "transpose=1,fps=2,scale=1920:-1"        -q:v 2        datasets/<NAME>/input/%06d.jpg  

> python convert.py   --source_path datasets/<NAME>   --camera OPENCV