% ----------------------------------------------
% setup.m
% 
% read mnist data into MATLAB, and restructure
%  into an array that can be converted to a tfrecords 
%  object
%
% Paul Kosek


% --------------- PART 0 ---------------
%  [initialization]


N_M = 28;                        % define number of points in row/col
imsize = N_M * N_M;              % number of bytes in an image
bytes_index_array = [1:imsize];  % static indexing array

image_file_offset = 16;          % offset from beginning of file
label_file_offset = 9;           % offset from beginning of file


test_points  = 10000;            % number of test items
train_points = 60000;            % number of train items


% --------------- PART 1 ---------------
%  [test data]

fid = fopen('t10k-images.idx3-ubyte');
raw_bytes = fread(fid);
fclose(fid);

fid = fopen('t10k-labels.idx1-ubyte');
lab = fread(fid);
fclose(fid);


test_x = zeros(test_points, N_M , N_M ); % preallocated structure - images
test_y = zeros(test_points, 1);          % preallocated structure - labels

% -- images -- 

for indx = 1:test_points

  test_x(indx,:,:) = ...
      permute( ...
        reshape( ...
          raw_bytes( imsize*(indx-1)+image_file_offset + bytes_index_array ), ...
          [N_M, N_M] ...
        )', ...
        [3, 1, 2] ...
      );
  
end

test_x = test_x./255; 

% -- labels -- 

label_file_offset = 9;            % offset from beginning of file

test_y( :, 1) = lab(label_file_offset:end); % assign


save('../data/mat/test', 'test_x', 'test_y');

% --------------- PART 2 ---------------
%  [train data]

fid = fopen('train-images.idx3-ubyte');
raw_bytes = fread(fid);
fclose(fid);

fid = fopen('train-labels.idx1-ubyte');
lab = fread(fid);
fclose(fid);


train_x = zeros(train_points, N_M , N_M ); % preallocated structure - images
train_y = zeros(train_points, 1);          % preallocated structure - labels

% -- images -- 

for indx = 1:train_points

  train_x(indx,:,:) = ...
      permute( ...
        reshape( ...
          raw_bytes( imsize*(indx-1)+image_file_offset + bytes_index_array ), ...
          [N_M, N_M] ...
        )', ...
        [3, 1, 2] ...
      );
  
end

train_x = train_x./255;             % scale data

% -- labels -- 

train_y( :, 1) = lab(label_file_offset:end); % assign

% -- split data -- 

fsize = 10000;

train_x0 = train_x(0*fsize + [1:fsize],:,:);
train_x1 = train_x(1*fsize + [1:fsize],:,:);
train_x2 = train_x(2*fsize + [1:fsize],:,:);
train_x3 = train_x(3*fsize + [1:fsize],:,:);
train_x4 = train_x(4*fsize + [1:fsize],:,:);
train_x5 = train_x(5*fsize + [1:fsize],:,:);

train_y0 = train_y(0*fsize + [1:fsize],1);
train_y1 = train_y(1*fsize + [1:fsize],1);
train_y2 = train_y(2*fsize + [1:fsize],1);
train_y3 = train_y(3*fsize + [1:fsize],1);
train_y4 = train_y(4*fsize + [1:fsize],1);
train_y5 = train_y(5*fsize + [1:fsize],1);

save( '../data/mat/train', 'train_x0', 'train_y0', 'train_x1', 'train_y1', 'train_x2', 'train_y2', 'train_x3', 'train_y3', 'train_x4', 'train_y4', 'train_x5', 'train_y5' )