import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


""" QUESTION 1 FUNCTIONS """

"""
display an image
function inspired from the documentation of opencv:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
"""
def ICV_show_img(img, label):
    w, h = int(img.shape[0]/2), int(img.shape[1]/2)
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(label, (h, w))
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

# perform rotation on an image
def ICV_rotate(img, angle_degrees):
    """
        :params img             :   image array
        :params angle_degrees   :   angle in degrees for rotation 
    """
    # convert to radians
    theta = np.radians(angle_degrees)

    # original dimensions
    old_x, old_y = img.shape[0], img.shape[1]

    # dimensions of new image after rotation
    new_x = round(abs(old_y*np.sin(theta)) + abs(old_x*np.cos(theta)))
    new_y = round(abs(old_y*np.cos(theta)) + abs(old_x*np.sin(theta)))

    # coordinates of the centre
    new_centre = (round(new_x/2), round(new_y/2))

    # initialise a new image array with the newly calculated dimensions
    new_img = np.empty((int(new_x), int(new_y), img.shape[2]), dtype=np.uint8)
    
    # rotation matrix
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    for i in range(int(new_x)):
        for j in range(int(new_y)):

            # translate to apply rotation by centre
            x = new_centre[0] - i
            y = new_centre[1] - j

            # perform backward mapping to obtain pixels
            new_img[i, j] = ICV_nn_interpolate(img, x, y, rot_matrix)
    return new_img

# nearest neighbor interpolation
def ICV_nn_interpolate(img, x, y, matrix):
    """
        :params img     :   image array
        :params x       :   scalar x-coordinate of the pixel
        :params y       :   scalar y-coordinate of the pixel
        :params matrix  :   rotation matrix
    """

    # apply rotation matrix
    x, y, _ = matrix @ np.array([x, y, 1])

    # coordinates of the centre
    centre = (np.round(img.shape[0]/2), np.round(img.shape[1]/2))

    # translate to the original coordinates
    x = centre[0] - x
    y = centre[1] - y

    x =  np.round(x)
    y =  np.round(y)

    # interpolate only within image boundaries
    # return black if mapped outside of original image boundaries
    if x >= img.shape[0]:
        return np.array([0, 0, 0], dtype=np.uint8)
    if x <= 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    if y >= img.shape[1]:
        return np.array([0, 0, 0], dtype=np.uint8)
    if y <= 0:
        return np.array([0, 0, 0], dtype=np.uint8)
        
    return img[int(x), int(y)]


# perform skewing on an image
def ICV_skew(img, angle_degrees):
    """
        :params img             :   image array
        :params angle_degrees   :   angle in degrees for skewing
    """

    # convert to radians
    theta = np.radians(angle_degrees)

    # dimensions of original image
    x, y = img.shape[0], img.shape[1]

    # new dimension size after skewing
    extra_y = np.ceil(x*np.tan(theta))
    skewed_y =  y + extra_y

    # initialise array with newly computed dimensions
    new_img = np.zeros((int(x), int(skewed_y), img.shape[2]), dtype=np.uint8)

    # skewing matrix
    skew_matrix = np.array([
        [1, 0, 0],
        [-np.tan(theta), 1, 0],
        [0, 0, 1]
    ])


    for i in range(x):
        for j in range(y):
            new_x, new_y, _ = skew_matrix @ np.array([i, j, 1])

            # translate horizontal axis by the difference of original dimension and new dimension
            new_img[int(new_x), int(new_y)+int(extra_y)] = img[i, j]
    return new_img


# Function to create an image with text
# Function inspired from:
# https://pythonprogramming.altervista.org/make-an-image-with-text-with-python/
def ICV_create_name(name, label):
    """
        :params name    :   string to place n the image
        :params label   :   path to save the image file
    """
    img = Image.new(mode='RGB', size=(256,256), color='yellow')
    font = ImageFont.truetype('arial.ttf', 72)

    draw = ImageDraw.Draw(img)
    draw.text((75,75), name, font=font, fill='black')
    img = np.array(img)
    cv2.imwrite('figures/transformations/{}.jpg'.format(label), img)
    return img



"""  QUESTION 2 FUNCTIONS """
# add border in the image by replicating the edges
def ICV_add_border(img):
    x, y = img.shape[0], img.shape[1]

    # Mirror the first row
    new_img = np.insert(img, 0, 0, axis=0)    
    new_img[0, :] = new_img[1, :]

    x = new_img.shape[0]
    # Mirror the bottom row
    new_img = np.insert(new_img, x-1, 0, axis=0)
    new_img[x-1, :] = new_img[x-2, :]

    # Mirror the first column
    new_img = np.insert(new_img, 0, 0, axis=1)
    new_img[:, 0] = new_img[:, 1]

    y = new_img.shape[1]
    # Mirror the last column
    new_img = np.insert(new_img, y-1, 0, axis=1)
    new_img[:, y-1] = new_img[:, y-2]
    return new_img

# remove border
def ICV_remove_border(img):

    # Remove the first and bottom rows
    new_img = np.delete(img, 0, axis=0)
    new_img = np.delete(new_img, new_img.shape[0]-1, axis=0)
    
    # Remove the leftmost and rightmost column
    new_img = np.delete(new_img, 0, axis=1)
    new_img = np.delete(new_img, new_img.shape[1]-1, axis=1)
    return new_img


# apply convolution
def ICV_apply_kernel(img, kernel):
    """
        :params img     :   image array
        :params kernel  :   3x3 numpy array of a kernel to be applied
    """
    # add border for smoothing edge pixels
    img = ICV_add_border(img)

    # dimensions of original image
    x, y = img.shape[0], img.shape[1]

    # apply on grayscale
    if img.shape[2] == 1:
        new_img = np.empty((x, y, 1), dtype=np.uint8)

        # apply kernel on a 3x3 window
        for i in range(1, x-1):
            for j in range(1, y-1):
                center = (kernel[1,1] * img[i, j]).astype(int)
                nw = (kernel[0,0] * img[i-1, j-1]).astype(int)
                n = (kernel[0,1] * img[i-1, j]).astype(int)
                ne = (kernel[0,2] * img[i-1, j+1]).astype(int)
                e = (kernel[1,2] * img[i, j+1]).astype(int)
                se = (kernel[2,2] * img[i+1, j+1]).astype(int)
                s = (kernel[2,1] * img[i+1, j]).astype(int)
                sw = (kernel[2,0] * img[i+1, j-1]).astype(int)
                w = (kernel[1,0] * img[i, j-1]).astype(int)
                
                result = center + nw + n + ne + e + se + s + sw + w
                pixel = result[0] if result[0] > 0 and result[0] < 255 else 255 if result[0] > 255 else 0
                new_img[i, j] = pixel

    # apply on RGB image          
    else:
        new_img = np.empty((x, y, 3), dtype=np.uint8)
        for i in range(1, x-1):
            for j in range(1, y-1):

                # apply kernel on a 3x3 window
                center = (kernel[1,1] * img[i, j]).astype(int)
                nw = (kernel[0,0] * img[i-1, j-1]).astype(int)
                n = (kernel[0,1] * img[i-1, j]).astype(int)
                ne = (kernel[0,2] * img[i-1, j+1]).astype(int)
                e = (kernel[1,2] * img[i, j+1]).astype(int)
                se = (kernel[2,2] * img[i+1, j+1]).astype(int)
                s = (kernel[2,1] * img[i+1, j]).astype(int)
                sw = (kernel[2,0] * img[i+1, j-1]).astype(int)
                w = (kernel[1,0] * img[i, j-1]).astype(int)

                # Find the mean of the sum
                b = center[0] + nw[0] + n[0] + ne[0] + e[0] + se[0] + s[0] + sw[0] + w[0]
                g = center[1] + nw[1] + n[1] + ne[1] + e[1] + se[1] + s[1] + sw[1] + w[1]
                r = center[2] + nw[2] + n[2] + ne[2] + e[2] + se[2] + s[2] + sw[2] + w[2]

                # threshold for color intensities
                r = r if r > 0 and r < 255 else 255 if r > 255 else 0
                g = g if g > 0 and g < 255 else 255 if g > 255 else 0
                b = b if b > 0 and b < 255 else 255 if b > 255 else 0
                new_img[i, j] = np.array([b, g, r], dtype=np.uint8)

    # remove the dummy border
    new_img = ICV_remove_border(new_img)
    return new_img

# convert to grayscale
def ICV_to_grayscale(img):
    x, y = img.shape[0], img.shape[1]

    new_img = np.empty((x, y, 1), dtype=np.uint8)

    # coefficients for each color
    r_c = 0.2126
    g_c = 0.7152
    b_c = 0.0722

    for i in range(0, x):
        for j in range(0, y):
            r = img[i, j][0]
            g = img[i, j][1]
            b = img[i, j][2]
            pixel = np.round((r_c * r) + (g_c * g) + (b_c * b))
            new_img[i, j] = int(pixel)

    return new_img




##################################### QUESTION 3 FUNCTIONS

# get frames of a video sequence
def ICV_get_frames(path, rbg):
    """
        :params path    :   path to the video to open
        :params rgb     :   boolean, 1 to load rgb video
    """
    vid = cv2.VideoCapture(path)

    if (vid.isOpened()== False): 
        print("Error: Unable to open video")

    # store all the frames of a video sequence
    frames = []

    while(vid.isOpened()):
        ret, frame = vid.read()

        if ret==True:
            if rbg==0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame[:, :, np.newaxis]
            # append all the retrieved frames
            frames.append(frame)
        else:
            break   
    vid.release()
    cv2.destroyAllWindows()
    return frames

# fill the count with 0 at indices between 0-255 of intensities that were not found
def ICV_fill_intensities(intensity, count):
    """
        :params intensity   :   unique intensity values
        :params count       :   count of each unique intensity value
    """
    intensity_filled = np.zeros((256), dtype=np.uint64)
    count_filled = np.zeros((256), dtype=np.uint64)

    intensity_filled[intensity] = intensity
    count_filled[intensity] = count

    return intensity_filled, count_filled



# create a histogram
def ICV_create_hist(img):
    """
        :params img     :   image array
        :params label   :   name of the figure to be saved on disk
    """

    # RGB images
    if img.ndim == 3:
        # Get each intensity and their corresponding count
        b, b_count = np.unique(img[:, :, 0], return_counts=True)
        g, g_count = np.unique(img[:, :, 1], return_counts=True)
        r, r_count = np.unique(img[:, :, 2], return_counts=True)

        # fill the count with 0 at indices between 0-255 of intensities that were not found
        b, b_count = ICV_fill_intensities(b, b_count)
        g, g_count = ICV_fill_intensities(g, g_count)
        r, r_count = ICV_fill_intensities(r, r_count)
        return ((b_count), (g_count), (r_count))

    # grayscale images
    else:
        intensity, count = np.unique(img[:, :], return_counts=True)
        intensity, count = ICV_fill_intensities(intensity, count)
        return count


# plot color histogram
def ICV_plot_histogram(h, label):
    """
        :params h       :   tuple containing the histogram of blue, green, red
        :params label   :   path to save the histogram
    """
    # x axis for each intensity value
    x = np.arange(0, 256)

    fig, ax = plt.subplots(3,1, figsize=(5,5))
    fig.tight_layout()

    bhist = ax[0].bar(x, h[0], color='b')
    ax[0].set_title('Blue Histogram')
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel('Intensity')

    ghist = ax[1].bar(x, h[1], color='g')
    ax[1].set_title('Green Histogram')
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel('Intensity')

    rhist = ax[2].bar(x, h[2], color='r')
    ax[2].set_title('Red Histogram')
    ax[2].set_ylabel('Count')
    ax[2].set_xlabel('Intensity')
    fig.show()
    fig.savefig(label)



# compute histogram intersection of two frames
def ICV_hist_intersection(h1, h2):
    """
        :params h1, h2                : intensity count of each histogram
        :params h1[0], h1[1], h[2]    : red, green, blue
    """

    # grayscale
    if type(h1) is not tuple:
        intersection = np.sum(np.minimum(h1, h2))
        return intersection
    # RGB
    else:
        b_intersection = np.sum(np.minimum(h1[0], h2[0]))
        g_intersection = np.sum(np.minimum(h1[1], h2[1]))
        r_intersection = np.sum(np.minimum(h1[2], h2[2]))
        return (b_intersection, g_intersection, r_intersection)


# normalize intersections
def ICV_normalize_intersections(intersections, pixels):
    """
        :params intersections   :   list of intersection values
        :params pixels          :   total number of pixels to normalize by
    """ 
    normalized = np.array(intersections, dtype=np.uint64)/pixels
    return normalized


# plot the intersection values
def ICV_plot_intersections(frames, intersections, plot, label):
    """
        :params frames          :   list of frames from a video sequence
        :params intersections   :   tuple containing intersection of each color
        :params plot            :   'bar' or 'line' for plot type
        :params label           :   string containing the path to be saved
    """
    fig, ax = plt.subplots(3,1)
    x = np.arange(1,len(frames))

    if plot == 'bar':
        ax[0].bar(x, intersections[0], color='b')
        ax[1].bar(x, intersections[1], color='g')
        ax[2].bar(x, intersections[2], color='r')
    else:
        ax[0].plot(x, intersections[0], color='b')
        ax[1].plot(x, intersections[1], color='g')
        ax[2].plot(x, intersections[2], color='r')
    fig.savefig(label)


# compute intersections of a video sequence
def ICV_get_intersections(frames, color):
    """
        :params frames      :   list of frames from a video sequence
        :params color       :   0, 1, 2, each representing blue, green, red respectively
    """
    intersections = []

    for i, frame in enumerate(frames):
        if i==len(frames)-1:
            break

        # get the histogram of two consecutive frames
        count_1 = ICV_create_hist(frames[i][:, :, color])
        count_2 = ICV_create_hist(frames[i+1][:, :, color])
        # compute intersection of the two histograms
        intersection = ICV_hist_intersection(count_1, count_2)
        intersections.append(intersection)
    return intersections


"""
function inspired by opencv documentation
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
"""
def ICV_play_video(frames):
    for frame in frames:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    pass


########################################### QUESTION 4


# split image into feature_descriptors
def ICV_to_windows(img, size):
    window_size = size
    x, y = img.shape[0], img.shape[1]

    windows = []

    for i in range(0, x,  window_size):
        for j in range(0, y, window_size):
            window = img[i:i+window_size, j:j+window_size]
            windows.append(window)
    return windows


# apply local binary pattern
def ICV_apply_lbp(img):
    x, y = img.shape[0], img.shape[1]
    new_img = np.zeros((x, y, 1), dtype=np.uint8)

    for i in range(1, x-1):
        for j in range(1, y-1):

            # Compare center pixel to neighborhood
            center = img[i, j].astype(int)
            nw = 0 if center > img[i-1, j-1] else 1
            n = 0 if center > img[i-1, j] else 1
            ne = 0 if center > img[i-1, j+1] else 1
            e = 0 if center > img[i, j+1] else 1
            se = 0 if center > img[i+1, j+1] else 1
            s = 0 if center > img[i+1, j] else 1
            sw = 0 if center > img[i+1, j-1] else 1
            w = 0 if center > img[i, j-1] else 1

            # Bit code of the neighbors
            binary_string = [ne, e, se, s, sw, w, nw, n]
            decimal = 0

            # Convert to decimal
            for k, bit in enumerate(binary_string):
                decimal += bit*(2**k)

            new_img[i, j] = decimal
    return new_img

def ICV_get_fd(img, window_size):
    img = ICV_to_grayscale(img)
    
    windows = ICV_to_windows(img, window_size)

    feature_descriptors = []
    lbp_windows = []

    for window in windows:
        new_window = ICV_add_border(window)
        lbp = ICV_apply_lbp(new_window)
        lbp_windows.append(lbp)
        histogram = ICV_create_hist(lbp[:, :, 0])
       
        feature_descriptors.append(histogram)
    return feature_descriptors, lbp_windows

def ICV_get_gd(feature_descriptors):
    feature_descriptors = np.array(feature_descriptors)
    for i, fd in enumerate(feature_descriptors):
        if i==0:
            gd = fd
        else:
            gd = np.concatenate((gd, fd), axis=None)
    return np.array(gd)


########################################## QUESTION 5

# 5C
def ICV_get_background(frames):
    frames = np.array(frames)
    x, y, z = frames.shape[1], frames.shape[2], frames.shape[3]

    avg = np.zeros((x, y, z), dtype=np.uint8)
    
    for i in range(x):
        for j in range(y):
            avg[i, j] = np.mean(frames[:, i, j]) if z==1 else np.mean(frames[:, i, j, :])
    return avg


# 5A
def ICV_compute_segmentation(frames, reference_frame):
    frames = np.array(frames)
    new_frames = np.abs(frames.astype('int16') - reference_frame.astype('int16'))
    new_frames = new_frames.astype('uint8')
    return new_frames

# 5B
def ICV_segment_previous_frame(frames):
    frames = np.array(frames)

    new_frames = []
    for i, frame in enumerate(frames):
        if i==0:
            new_frames.append(frame)
            continue
        new_frame = np.abs(frame.astype('int16') - frames[i-1].astype('int16'))
        new_frames.append(new_frame)
    new_frames = np.array(new_frames, dtype=np.uint8)
    return new_frames

def ICV_threshold_classification(frames, threshold):
    frames = np.array(frames, dtype=np.uint8)
    frames[frames < threshold] = 0
    frames[frames >= threshold] = 255
    return frames

# 5D
def ICV_fill_blobs(img):
    new_img = img.copy()

    white_indices = np.argwhere(img[:, :, 0]==255)
    x_indices = white_indices[:, 0]
    y_indices = white_indices[:, 1]

    for x, y in zip(x_indices, y_indices):
        if x < img.shape[0]-2 and y < img.shape[1]-2:
            if img[x, y+1] == 0:
                new_img[x, y+1] = 255
            if img[x, y-1] == 0:
                new_img[x, y-1] = 255
            if img[x+1, y] == 0:
                new_img[x+1, y] = 255
            if img[x+1, y+1] == 0:
                new_img[x+1, y+1] = 255
            if img[x+1, y-1] == 0:
                new_img[x+1, y-1] = 255
            if img[x-1, y] == 1:
                new_img[x-1, y] = 255
            if img[x-1, y+1] == 0:
                new_img[x-1, y+1] = 255
            if img[x-1, y-1] == 0:
                new_img[x-1, y-1] = 255
    return new_img
# COMPUTATIONAL IMPRVOEMET WITH LOOPING ONLY WHITE INSTEAD OF ALL PIXELS

def ICV_interpolate_frames(frames, interpolate_level):
    interpolated_frames = []
    for frame in frames:
        interpolated = frame.copy()
        for i in range(interpolate_level):
            interpolated = ICV_fill_blobs(interpolated)
        interpolated_frames.append(interpolated)
    return interpolated_frames