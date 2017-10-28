# Snapchat Lens Effect in Python

This is the companion repository for the blog post at [makeartwithpython.com](https://www.makeartwithpython.com/blog/building-a-snapchat-lens-effect-in-python/).

## Architecture

![Lens Architecture](https://github.com/burningion/snapchat-lens-effect-in-python/raw/master/images/eyeflow.png)

## Usage

You'll need to download the [shape_predictor_68](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) from dlib-models and unzip it in this directory first.

After, you should be able to just pass in the location of that predictor to the Python3 program as a command line argument like so:

```bash
$ python3 eye-glitch.py -predictor shape_predictor_68_face_landmarks.dat 
```

... and by pressing 's' to enable the eye snake, you'll end up with something like this:

![Image like this](https://github.com/burningion/snapchat-lens-effect-in-python/raw/master/images/out.gif)

See the full post at https://www.makeartwithpython.com/blog/building-a-snapchat-lens-effect-in-python/
