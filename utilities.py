import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# display an image
def show_img(img, label):
    w, h = int(img.shape[0]/2), int(img.shape[1]/2)
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(label, (h, w))
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

# perform rotation on an image
def rotate(img, angle_degrees):
    """
        :params img             :   image array
        :params angle_degrees   :   angle in degrees for rotation 
    """

    # convert to radians
    theta = np.radians(angle_degrees)

    old_x, old_y = img.shape[0], img.shape[1]

    # dimensions of new image after rotation
    new_x = round(abs(old_y*np.sin(theta)) + abs(old_x*np.cos(theta)))
    new_y = round(abs(old_y*np.cos(theta)) + abs(old_x*np.sin(theta)))
    new_centre = (round(new_x/2), round(new_y/2))

    new_img = np.empty((int(new_x), int(new_y), img.shape[2]), dtype=np.uint8)
    
    # romation matrix
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
            new_img[i, j] = nn_interpolate(img, x, y, rot_matrix)
    return new_img

# nearest neighbor interpolation
def nn_interpolate(img, x, y, matrix):
    """
        :params img     :   image array
        :params x       :   x-coordinate of the pixel
        :params y       :   y-coordinate of the pixel
        :params matrix  :   rotation matrix
    """

    # apply rotation matrix
    x, y, _ = matrix @ np.array([x, y, 1])

    centre = (np.round(img.shape[0]/2), np.round(img.shape[1]/2))

    # translate to the original coordinates
    x = centre[0] - x
    y = centre[1] - y

    x =  np.round(x)
    y =  np.round(y)

    # interpolate only within image boundaries
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
def skew(img, angle_degrees):
    """
        :params img             :   image array
        :params angle_degrees   :   angle in degrees for skewing
    """

    # convert to radians
    theta = np.radians(angle_degrees)

    x, y = img.shape[0], img.shape[1]

    # new dimension size after skewing
    extra_y = np.ceil(x*np.tan(theta))
    skewed_y =  y + extra_y

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
            new_img[int(new_x), int(new_y)+int(extra_y)] = img[i, j]
    return new_img


def create_name(name, label):
    img = Image.new(mode='RGB', size=(256,256), color='yellow')
    font = ImageFont.truetype('arial.ttf', 72)

    draw = ImageDraw.Draw(img)
    draw.text((75,75), name, font=font, fill='black')
    img = np.array(img)
    cv2.imwrite('figures/transformations/{}.jpg'.format(label), img)
    return img

# add border in the image by replicating the edges
def add_border(img):
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
def remove_border(img):

    # Remove the first and bottom rows
    new_img = np.delete(img, 0, axis=0)
    new_img = np.delete(new_img, new_img.shape[0]-1, axis=0)
    
    # Remove the leftmost and rightmost column
    new_img = np.delete(new_img, 0, axis=1)
    new_img = np.delete(new_img, new_img.shape[1]-1, axis=1)
    return new_img


def apply_kernel(img, kernel):
    """
        :params img     :   image array
        :params kernel  :   3x3 numpy array of a kernel to be applied
    """
    # add border for smoothing edge pixels
    img = add_border(img)

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
                r = center[0] + nw[0] + n[0] + ne[0] + e[0] + se[0] + s[0] + sw[0] + w[0]
                g = center[1] + nw[1] + n[1] + ne[1] + e[1] + se[1] + s[1] + sw[1] + w[1]
                b = center[2] + nw[2] + n[2] + ne[2] + e[2] + se[2] + s[2] + sw[2] + w[2]

                # threshold for color intensities
                r = r if r > 0 and r < 255 else 255 if r > 255 else 0
                g = g if g > 0 and g < 255 else 255 if g > 255 else 0
                b = b if b > 0 and b < 255 else 255 if b > 255 else 0
                new_img[i, j] = np.array([r, g, b], dtype=np.uint8)
    new_img = remove_border(new_img)
    return new_img


def to_grayscale(img):

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

# fill the count array
def fill_intensities(intensity, count):
    intensity_filled = np.zeros((256), dtype=np.uint8)
    count_filled = np.zeros((256), dtype=np.uint8)

    intensity_filled[intensity] = intensity
    count_filled[intensity] = count

    return intensity_filled, count_filled

# create a histogram
def create_hist(img, label):
    """
        :params img     :   image array
        :params label   :   name of the figure to be saved on disk
    """

    # RGB images
    if img.shape[2] == 3:

        # Get each intensity and their corresponding count
        r, r_count = np.unique(img[:, :, 0], return_counts=True)
        g, g_count = np.unique(img[:, :, 1], return_counts=True)
        b, b_count = np.unique(img[:, :, 2], return_counts=True)

        # fill the count with 0 if intensity not found
        r, r_count = fill_intensities(r, r_count)
        g, g_count = fill_intensities(g, g_count)
        b, b_count = fill_intensities(b, b_count)

        fig, ax = plt.subplots(3,1, figsize=(5,5))
        fig.tight_layout()

        rhist = ax[0].bar(r, r_count)
        ax[0].set_title('Red Histogram')
        ax[0].set_ylabel('Count')

        ghist = ax[1].bar(g, g_count)
        ax[1].set_title('Green Histogram')
        ax[1].set_ylabel('Count')

        bhist = ax[2].bar(b, b_count)
        ax[2].set_title('Blue Histogram')
        ax[2].set_ylabel('Count')
        fig.show()
        fig.savefig(label)
        return ((r_count), (g_count), (b_count))

    # grayscale images
    else:
        intensity, count = np.unique(img[:, :], return_counts=True)

        intensity, count = fill_intensities(intensity, count)

        count = count/img.size

        fig, ax = plt.subplots(1,1)
        fig.tight_layout()

        hist = ax.bar(intensity, count)
        ax.set_title('Intensity')
        ax.set_ylabel('Count')
        return count

def hist_intersection(h1, h2):
    """
        :param h1, h2                : intensity count of each histogram
        :param h1[0], h1[1], h[2]    : red, green, blue
    """

    # grayscale
    if type(h1) is not tuple:
        intersection = np.sum(np.minimum(h1, h2))
        print('Histogram Intersection: ', intersection)
        return intersection
    # RGB
    else:
        r_intersection = np.sum(np.minimum(h1[0], h2[0]))
        g_intersection = np.sum(np.minimum(h1[1], h2[1]))
        b_intersection = np.sum(np.minimum(h1[2], h2[2]))

        print('Red Intersection: ', r_intersection)
        print('Green Intersection: ', g_intersection)
        print('Blue Intersection: ', b_intersection)
        return (r_intersection, g_intersection, b_intersection)

def get_frames(path, rbg):
    vid = cv2.VideoCapture(path)

    if (vid.isOpened()== False): 
        print("Error opening video stream or file")

    frames = []

    while(vid.isOpened()):
        ret, frame = vid.read()

        if ret==True:
            if rbg==0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame[:, :, np.newaxis]
            frames.append(frame)
        else:
            break   
    vid.release()
    cv2.destroyAllWindows()
    return frames

def play_video(frames):
    for frame in frames:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    pass