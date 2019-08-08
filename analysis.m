Imgs = zeros(256,256,30);
truth_Imgs =  zeros(256,256,30);
for i = 1:30
    Imgs(:,:,i) = im2double(imread([num2str(i-1),'.png_predict.png']));
end

for i = 1:30
   temp = im2double(imread(['.\label\',num2str(i-1),'.png'])); 
   truth_Imgs(:,:,i) = imresize(temp, 0.5) >= 0.5;
   
end

Threshold = zeros(256,256,30);
for i = 1:30
   X = reshape(Imgs(:,:,i),256*256,1);
   Y = kmeans(X,2);
   Z = reshape(Y,256,256);
   Threshold(:,:,i) = Z-1;
   num_zeros = sum(sum(Threshold(:,:,i)==0));
   num_ones = sum(sum(Threshold(:,:,i)==1));
   if num_zeros > num_ones
       Threshold(:,:,i) = -Threshold(:,:,i) + 1;
   end
   
%    figure;
%    imshow(Threshold(:,:,i));
end

IOU = zeros(30);

for i = 1:30
    IOU(i) = sum(sum(truth_Imgs(:,:,i) == Threshold(:,:,i)))/(256*256);
end

myHist = reshape(Imgs,256*256*30,1);
R = hist(myHist,10)/(256*256);
R = [0,R,0]
figure; plot(-0.05:0.1:1.05,R);
xlabel('pixel intensity');
ylabel('probability of obtaining that intensity');
title('histogram of intensity observed');

figure; plot(1:30,IOU);
xlabel('index of predicted/ground truth label');
ylabel('IOU between predicted and ground truth label')
title('IOU matrix for test images');

   