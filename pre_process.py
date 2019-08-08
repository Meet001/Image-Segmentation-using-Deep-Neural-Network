import Augmentor
import sys

p = Augmentor.Pipeline("data/original_images")

p.ground_truth("data/original_labels")

p.greyscale(1.0)

p.rotate90(0.15)
p.rotate90(0.15)
p.rotate90(0.15)


p.rotate(probability=0.5,max_left_rotation=15,max_right_rotation=15)

p.flip_random(0.3)

p.gaussian_distortion(0.15, 7, 7, 8,'bell','in')
p.gaussian_distortion(0.15, 5, 5, 8,'bell','in')
p.gaussian_distortion(0.15, 3, 3, 8,'bell','in')

p.crop_random(0.15, 0.9)
p.crop_random(0.15, 0.9)
p.crop_random(0.15, 0.9)

p.resize(1.0, 256, 256)

p.sample(int(sys.argv[1]))