from os import chdir as cd
from os import listdir as ls
from PIL import Image

cd('C:\\Users\\Geoffrey\\Documents\\GitHub\\MLproject')
imlist = ls('Leaves')

def normswap(img):
    """
    flips axes so that the height is greater than or equal to the width
    
    args: img: a PIL Image object
    returns: a PIL Image object
    """
    if img.height < img.width: return img.transpose(Image.ROTATE_90)
    else: return img

for imname in imlist:
    img = Image.open('Leaves\\'+imname)
    img = normswap(img)
    img = img.resize((60,90))
    img.save('PreprocessedLeaves\\'+imname[:-3]+'png')
    img.close()
    print('.', sep='', end='')
