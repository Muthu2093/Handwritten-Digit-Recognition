%% Function to read the MNIST testing images and labels and stores them in mnistTEST.mat file 

testImg = fopen('t10k-images-idx3-ubyte','r','b'); % first we have to open the binary file
MagicNumber = fread(testImg,1,'int32');
MagicNumber = 2051;
nImages = fread(testImg,1,'int32');% Read the number of images
nImages = 60000;
nRows = fread(testImg,1,'int32');% Read the number of rows in each image

nRows = 28;
nCols= fread(testImg,1,'int32');% Read the number of columns in each image
nCols = 28;
fseek(testImg,16,'bof');
for iddd=1:10000
    
    
    img= fread(testImg,28*28,'uchar');% each image has 28*28 pixels in unsigned byte format

    img2=zeros(28,28);
    for i=1:28
        img2(i,:)=img((i-1)*28+1:i*28);
    end
    %imshow(img2);
    
    img1(:,:)=img2(5:24,5:24);%Trim the extracted image
    img=img1./255;
    test_Images{iddd}=img;
end



%% Reading labels


    testlbl = fopen('t10k-labels-idx1-ubyte','r','b'); % first we have to open the binary file
    MagicNumber = fread(testlbl,1,'int32');
    MagicNumber = 2049;
    nLabels = fread(testlbl,1,'int32');% Read the number of labels
    nLabels = 60000;
fseek(testlbl,8,'bof');
test_labels=zeros(10,10000);
for i=1:10000
    aaaa=fread(testlbl,1,'uchar');
    if aaaa==0
        test_labels(10,i)=1;
        continue;
    end
    test_labels(aaaa,i)=1;
    
    %for i=1:28
    %    img2(i,:)=img((i-1)*28+1:i*28);
    %end
    %imshow(img2);

end

save('mnistTest.mat','test_Images','test_labels');


