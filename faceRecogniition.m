inputDir='C:\Users\DELL PC\Documents\MATLAB\trainingSet'; %image DB
imageDim=[399,399];
filenames=dir(fullfile(inputDir,'*.jpg'));
numImages=numel(filenames);
images=[];%blank matrix

for n=1:numImages
    filename=fullfile(inputDir,filenames(n).name);
    img=imread(filename);   
    img1=rgb2gray(img);
    img2=imresize(img1,[399 399]);
    if n==1
        images=zeros(prod(imageDim),numImages);  
    end
    images(:,n)=img2(:);  
end

%training
meanFace=mean(images,2); %mean of each row i.e each image
%disp('mean face');
%disp(meanFace);
shiftedImages=images-repmat(meanFace,1,numImages);

%cal eigen value
[evectors,score,evalues]=pca(images'); %principal component analysis
%feature vector from eigen vector
features=evectors'*shiftedImages;
%disp('features');
%disp(features);

%take input
x=videoinput('winvideo');
image=getsnapshot(x);
FaceDetector=vision.CascadeObjectDetector();
Boundary=step(FaceDetector,image);
B=insertObjectAnnotation(image,'rectangle',Boundary,'Homo Sapien');
figure,imshow(B),title('Detected Faces');
for i=1:size(Boundary,1)
C= imcrop(B,Boundary(i,:));
figure(1),subplot(2,3,i);imshow(C);
C=rgb2gray(C);
    C=imresize(C,[399 399]);
imwrite(C,'testfile1.jpg' );
end

%find similarity
featureVec=evectors'*(double(C(:))-meanFace);
disp(featureVec);
similarityScore=arrayfun(@(n)1/(1+norm(features(:,n)-featureVec)),1:numImages);

%sort 
[sortedX,sortingIndices]=sort(similarityScore,'descend');
maxValue=sortedX(1:5);
maxValueIndices=sortingIndices(1:5);


for x=1:5
figure,imshow([C reshape(images(:,maxValueIndices(x)),imageDim)]);
title(sprintf('match percent %f',maxValue(x)));
%to find the image
name=filenames(maxValueIndices(x)).name;
imageName=strsplit(name,'-');
disp(imageName);
n=strcat(imageName(1),'.jpg');
figure,imshow(imread(n{1}));

end



