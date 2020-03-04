import os
import shutil
import glob

def get_pov_id(frame_id):
    """
    Get pov id from frame id.
    """
    num_povs = 104
    pov_id = (frame_id % num_povs) - 1
    if pov_id < 0:
        pov_id += num_povs
    return pov_id

def mkdir(directory):
    """
    Create directory in file system, if it does not exists.
    :param directory: path to directory that will be created
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
NUM_OF_POVS = 104
images_gb = os.listdir('c0/gb')
images_nb = os.listdir('c0/nb')
images_pb = os.listdir('c0/pb')
labels = os.listdir('Labels')
segmaps = os.listdir('SegMaps')
for i in range(NUM_OF_POVS):
    mkdir('c0/gb/p' + str(i))
    mkdir('c0/nb/p' + str(i))
    mkdir('c0/pb/p' + str(i))
    mkdir('Labels/p' + str(i))
    mkdir('SegMaps/p' + str(i))

for image in images_gb:
    if image.endswith('.png'):
        a = image.split('_')
        pov_id = get_pov_id(int(a[0]))
        shutil.move('c0/gb/' + image, 'c0/gb/p' + str(pov_id))
for image in images_nb:
    if image.endswith('.png'):
        a = image.split('_')
        pov_id = get_pov_id(int(a[0]))
        shutil.move('c0/nb/' + image, 'c0/nb/p' + str(pov_id))
for image in images_pb:
    if image.endswith('.png'):
        a = image.split('_')
        pov_id = get_pov_id(int(a[0]))
        shutil.move('c0/pb/' + image, 'c0/pb/p' + str(pov_id))
for image in labels:
    if image.endswith('.xml'):
        a = image.split('.')
        pov_id = get_pov_id(int(a[0]))
        shutil.move('Labels/' + image, 'Labels/p' + str(pov_id))
for image in segmaps:
    if image.endswith('.png'):
        a = image.split('.')
        pov_id = get_pov_id(int(a[0]))
        shutil.move('SegMaps/' + image, 'SegMaps/p' + str(pov_id))
