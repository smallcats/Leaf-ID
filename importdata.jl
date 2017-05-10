#=
Script for importing picture data
=#

using FileIO: load
using Images: channelview

"""
Loads directories.

loaddirs(path=\".\") --> dirnames
"""
function loaddirs(path=".")
  dirnames = []
  dirlist = readdir(path)
  for dirname in dirlist
    dirpath = join([path, "\\", dirname])
    if isdir(dirpath)
      dirnames = vcat(dirnames, dirname)
    end
  end
  dirnames
end


"""
Loads an array of all image files

loadfilenames(path=\".\") --> filenames
"""
function loadfilenames(path=".")
  imfiles = []
  filelist = readdir(path)
  for filename in filelist
    if ismatch(r"\.jpg$|\.png$", filename)
      imfiles = vcat(imfiles, filename)
    end
  end
  imfiles
end


"""
Loads an image as a 3d array with elements of type N0f16.

loadimg(path) --> image

size(image) = (3,width,height)
"""
function loadimg(path)
  channelview(load(path))
end


""" Tested
Normalize an image given as a size (3,height,width) array to a size (3, newsize...) array with entries of type Float64.

normimg(img, newsize) --> newimg

Newsize must preserve aspect ratio (up to some error proportional to the 1/(scaling factor))!
"""
function normimg(img, newsize)
  #rotate to height<=width
  if size(img, 2)>size(img,3)
    img = permutedims(img, [1,3,2])
  end
  #check aspect ratio
  steps = [Int64(ceil(size(img,k)/newsize[k-1])) for k=2:3]
  if steps[1] == steps[2]
    step = steps[1]
  else
    throw(error("Aspect ratio was not preserved."))
  end
  newimg = Array{Float64,3}(3,newsize...)
  #image bulk:
  for i = 1:3
    for j = 1:newsize[1]-1
      for k = 1:newsize[2]-1
        newimg[i,j,k] = sum(img[i,step*(j-1)+1:step*j,step*(k-1)+1:step*k])/(step^2)
      end
    end
  end
  #right edge:
  roverhang = size(img,3) - step*(newsize[2]-1)
  for i = 1:3
    for j = 1:newsize[1]-1
      newimg[i,j,end] = sum(img[i,step*(j-1)+1:step*j, step*(newsize[2]-1)+1:end])/(roverhang*step)
    end
  end
  #bottom edge:
  boverhang = size(img,2) - step*(newsize[1]-1)
  for i = 1:3
    for k = 1:newsize[2]-1
      newimg[i,end,k] = sum(img[i,step*(newsize[1]-1)+1:end, step*(k-1)+1:step*k])/(step*boverhang)
    end
  end
  #bottom right corner
  for i = 1:3
    newimg[i,end,end] = sum(img[i,step*(newsize[1]-1)+1:end, step*(newsize[2]-1)+1:end])/(roverhang*boverhang)
  end
  newimg
end


"""
Gets a newsize image from path. The image is returned as a (3, height,width) array of Float64's.

getimg(path,newsize) --> img
"""
function getimg(path,newsize)
  normimg(loadimg(path),newsize)
end

#Constants
path = "leafsnap-dataset\\dataset\\images\\field"
newsize = (60,80)

#To be loaded
photos = Dict{String, Array{Array{Float64,3},1}}()

#Loading
let species = loaddirs(path), spnum = 3
  for sp = 1:spnum
    samplenames = loadfilenames(join([path,"\\",species[sp]]))
    samples = Array{Array{Float64,3},1}(length(samplenames))
    for s=1:length(samplenames)
      print("\rSpecies $(sp), Sample $s")
      samples[s] = getimg(join([path, "\\", species[sp], "\\", samplenames[s]]),newsize)
    end
    print("\n")
    photos[species[sp]] = samples
  end
end
