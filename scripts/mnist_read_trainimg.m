%% Function to read the MNIST training images and labels and stores them in mnist.mat file 

trainImg = fopen('train-images-idx3-ubyte','r','b'); % first we have to open the binary file
MagicNumber = fread(trainImg,1,'int32');
MagicNumber = 2051;
nImages = fread(trainImg,1,'int32');% Read the number of images
nImages = 60000;
nRows = fread(trainImg,1,'int32');% Read the number of rows in each image

nRows = 28;
nCols= fread(trainImg,1,'int32');% Read the number of columns in each image
nCols = 28;
fseek(trainImg,16,'bof');
for iddd=1:60000
      
    img= fread(trainImg,28*28,'uchar');% each image has 28*28 pixels in unsigned byte format

    img2=zeros(28,28);
    for i=1:28
        img2(i,:)=img((i-1)*28+1:i*28);
    end
    %imshow(img2);
    
    img1(:,:)=img2(5:24,5:24);%Trim the extracted image
    img=img1./255;
    train_Images{iddd}=img;
end

%% Reading labels

    trainlbl = fopen('train-labels-idx1-ubyte','r','b'); % first we have to open the binary file
    MagicNumber = fread(trainlbl,1,'int32');
    MagicNumber = 2049;
    nLabels = fread(trainlbl,1,'int32');% Read the number of labels
    nLabels = 60000;
fseek(trainlbl,8,'bof');
train_lables=zeros(10,60000);
for i=1:60000
    aaaa=fread(trainlbl,1,'uchar');
    if aaaa==0
        train_labels(10,i)=1;
        continue;
    end
    train_labels(aaaa,i)=1;
    
    %for i=1:28
    %    img2(i,:)=img((i-1)*28+1:i*28);
    %end
    %imshow(img2);

end
save('mnist.mat','train_Images','train_labels');



