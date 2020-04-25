s='g.jpg';
image=imread(s);
imageName=strsplit(s,'.');
fileName=imageName{1};
FaceDetector=vision.CascadeObjectDetector();
Boundary=step(FaceDetector,image);
B=insertObjectAnnotation(image,'rectangle',Boundary,'Homo Sapien');
figure,imshow(B),title('Detected Faces');

for i=1:size(Boundary,1)
C= imcrop(B,Boundary(i,:));
figure(1),subplot(2,3,i);
figure,imshow(C);
%C=rgb2gray(C);
    C=imresize(C,[400 400]);
imwrite(C,sprintf('%s-%d.jpg',fileName,i));
end