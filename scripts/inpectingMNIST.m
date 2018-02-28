%% This function inspects the image in the MNIST data set

load mnist.mat 
figure(1);
imshow(train_Images{1})
filterBank=createFilterBank();

%% Filtering the images
for i=1:1:5
    for idx=1:5:20
     a{idx}=imfilter(train_Images{i},filterBank{idx});
     a{idx}=imresize(a{idx},10,'bilinear');
     if(idx==1) 
            filterResponses1=a{idx};
           filterResponses=a{idx};
       end
        if(idx>1)
            filterResponses1=cat(4,filterResponses1,a{idx});
            filterResponses=cat(3,filterResponses,a{idx});
        end
    end
    figure(i+1);
    montage(filterResponses1, 'Size',[4 NaN]);
end


