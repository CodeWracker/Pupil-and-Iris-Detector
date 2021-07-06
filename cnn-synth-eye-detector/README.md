# Eye feature border estimator

## Data collected

I used the dataset provided by the cambrige college (https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/) and extracted the following:

- Center of the iris (and the pupil, consequently)
- Radius of the circles that characterizes the iris and the pupil
- The image, converted in grey scale, and flattened into a (9600,) numpy array
  > The original image had the shape of (80,120,3), the, after conversion it had (80,120)
