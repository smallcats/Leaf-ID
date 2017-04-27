#Script for loading and putting data in a usable format. Divides data into three
#matrices, X, for training, X_cv for cross-validation, and X_test for testing,
#along with y, y_cv and y_test.
#Each sample inhabits a row of X, unrolled.

clear all;

pkg load image

#Constants
dirpath = 'leafsnap-dataset/dataset/images/field';
num_sp = 10;

#Load leaf paths
species = readdir(dirpath)(3:end);
leaves = cell(num_sp,1);
num_leaves = zeros(num_sp,1);
leaf_ind = zeros(num_sp,1);
fprintf('\n');
for sp = 1:num_sp
  tempcell = readdir(strcat(dirpath,'/',species{sp}))(3:end);
  for filenum = 1:length(tempcell)
    if tempcell{filenum}(end-3:end) == '.jpg'
      leaves{sp}{end+1} = tempcell{filenum};
    endif
  endfor
  num_leaves(sp) = length(leaves{sp});
  leaf_ind(sp:end) += num_leaves(sp);
  fprintf('\rSpecies loaded: %d', sp); 
endfor

leaf_ind = [0;leaf_ind];
tot_leaves = sum(num_leaves);

#Load images
leafimgs = cell(tot_leaves,1);
sp = 1;
fprintf('\n');
for leaf = 1:tot_leaves
  #find species of leaf
  if leaf > leaf_ind(sp+1)
    sp += 1;
  endif
  #load leaf image
  fprintf('\rLoading leaf image %d from species %d', leaf,sp)
  leafimgs{leaf} = imread(strcat(dirpath,'/',species{sp},'/',leaves{sp}{leaf-leaf_ind(sp)}));
endfor

#clear unused variables. Only leafimgs and num_leaves remain.
clear leaf dirpath leaves num_leaves sp species filenum tempcell;

#clean dataset - ensure all imgs have the same size, and decrease size of data
newsize = [30,40];
for k = 1:tot_leaves
  if size(leafimgs{k})(1) > size(leafimgs{k})(2)
    leafimgs{k} = permute(leafimgs{k},[2,1,3]);
  endif
  leafimgs{k} = imresize(leafimgs{k}, newsize);
endfor

#unroll into X
X = zeros(tot_leaves, 3600);
for k = 1:tot_leaves
  X(k,:) = leafimgs{k}(:)';
endfor

#Randomize
perm = randperm(tot_leaves);
X = X(perm,:);

#Normalize
X /= max(max(X))/2;
X -= mean(mean(X));

step = int8(tot_leaves/9);

X_cv = X(1:step,:);
X_test = X(step+1:2*step,:);
X = X(2*step+1:end,:);

#Labels
y = ones(tot_leaves,1);
for k = 2:num_sp
  y(leaf_ind(k)+1:leaf_ind(k+1)) *= k;
endfor
y -= 1;
y = y(perm);
y_cv = y(1:step);
y_test = y(step+1:2*step);
y = y(2*step+1:end);

clear leaf_ind perm tot_leaves num_sp k leafimgs new_size change_size old_size step newsize;
