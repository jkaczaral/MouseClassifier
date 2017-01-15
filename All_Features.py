import os
import sys
import collections as col
import numpy as np
from skimage import io as io
from skimage.color import rgb2gray
import mahotas as mh
import mahotas.features

#user input to be analyzed
arg1 = sys.argv[1]

#get image names into list from file
def get_images(filename):
	with open(filename, 'r') as images:
		image_names = []
		for line in images:
			line = line.rstrip('\n')
			line = line.split(',',1)
			name = line[0]

			image_names.append(name)

	return image_names


#get images into array, convert to grayscale and rbgfloat
def img2array(image_names):
	grey_mice = []
	rgb_mice = []
	for image in image_names:
		image = image + '.jpg'
		filename = os.path.join(image)
		mouse = io.imread(filename)
		rgb_mice.append(mouse)
		mouse_gray = rgb2gray(mouse)
		grey_mice.append(mouse_gray)

	return grey_mice, rgb_mice


#mask values > 0.75 to eliminate white background, 
#add those masked images to a list, 
def mask_mouse(grey_mice):
	masked_mice = []
	for mouse in grey_mice:
		masked_mouse = np.ma.masked_greater_equal(mouse,0.75)
		masked_mice.append(masked_mouse)
	return masked_mice


#Stats to a dictonary for grey and rgb
def grey_features(masked_mice):
	features_all_mice_grey = []
	for mouse in masked_mice:
		stats = []
		mean = np.mean(mouse)
		stats.append(repr(mean))
		std = np.std(mouse)
		stats.append(repr(std))
		var = np.var(mouse)
		stats.append(repr(var))
		count = mouse.count()
		percent_masked = (float(count))/ float(307200)
		stats.append(repr(percent_masked))
		lbp_features = mh.features.lbp(mouse, 4, 8)
		lbp_hist = np.array_str(lbp_features, max_line_width=5000)
		hist  = np.asarray(lbp_features)
		mean_hist = np.mean(hist)
		stats.append(repr(mean_hist))
		stats.append(lbp_hist)
		stats = ','.join(stats)
		features_all_mice_grey.append(stats)
	features_grey_mice = dict(zip(image_names, features_all_mice_grey))
		
	return features_grey_mice

def rgb_features(rgb_mice):
	
	features_all_mice = []
	for mouse in rgb_mice:
		features = []
		har_features = mahotas.features.haralick(mouse).mean(0)
		har_features = np.array_str(har_features, max_line_width=5000)
		features.append(har_features)
		pftas_features = mahotas.features.pftas(mouse)
		pftas_features = np.array_str(pftas_features, max_line_width=5000)
		features.append(pftas_features)		
		features = ','.join(features)
		features_all_mice.append(features)
	features_rgb_mice = dict(zip(image_names, features_all_mice))
		
	return features_rgb_mice

#actually do the things
image_names = get_images(arg1)
grey_mice, rgb_mice = img2array(image_names)
masked_mice = mask_mouse(grey_mice)
features_grey_mice = grey_features(masked_mice)
features_rgb_mice = rgb_features(rgb_mice)

#sort dictonary to be in numerical order for writing to file
features_grey_mice = col.OrderedDict(sorted(features_grey_mice.items()))
features_rgb_mice = col.OrderedDict(sorted(features_rgb_mice.items()))



#writes stats to a file
with open('All_Features.txt', 'w') as outfile:
	for key, value in features_grey_mice.iteritems():
		outfile.write( key + ',' + str(features_grey_mice[key]) + ',' + str(features_rgb_mice[key]) +'\n')

print 'All done!'
