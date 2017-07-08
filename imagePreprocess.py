from os import chdir as cd
from os import listdir as ls
from os import mkdir
from os.path import isdir
from PIL import Image

cd('C:\\Users\\Geoffrey\\Documents\\GitHub\\MLproject')

def lspath(path):
    """
    lists the files in a directory, with path relative to current directory

    args: path: the relative path to a directory
    """
    filenames = []
    for name in ls(path):
        filenames.append(path+'\\'+name)
    return filenames

def getfilenames(path):
    filequeue = lspath(path)
    filenames = []
    dirnames = []
    while filequeue != []:
        filename = filequeue.pop()
        if isdir(filename):
            filequeue.extend(lspath(filename))
            dirnames.append(filename)
        elif filename[-4:] == '.jpg': filenames.append(filename)
        dirnames.sort(key=(lambda x: len(x)))
    return filenames, dirnames

filenames, dirnames = getfilenames('.\\leafsnap-dataset\\dataset\\images\\lab')

def normswap(img):
    """
    flips axes so that the height is greater than or equal to the width
    
    args: img: a PIL Image object
    returns: a PIL Image object
    """
    if img.height < img.width: return img.transpose(Image.ROTATE_90)
    else: return img

for dirname in dirnames:
    mkdir('PreprocessedLeaves\\leafsnap-lab\\'+dirname[38:])

for k, filename in enumerate(filenames):
    img = Image.open(filename)
    img = normswap(img)
    img = img.resize((60,90))
    img.save('PreprocessedLeaves\\leafsnap-lab\\'+filename[38:-3]+'png')
    img.close()
    if k%10 == 0: print('.', sep='', end='')
