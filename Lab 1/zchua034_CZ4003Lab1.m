% CZ4003 Lab 1
%% 2.1 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.1 Contrast Stretching
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% a. Input the image into a MATLAB matrix variable by executing:
%-----------------------------------------------------------------
Pc = imread('mrt-train.jpg'); 
whos Pc ;
P=rgb2gray(Pc);
whos P;

%-----------------------------------------------------------------
% b. View this image using imshow
%-----------------------------------------------------------------
imshow(Pc);
imshow(P);

%-----------------------------------------------------------------
% c.Check the minimum and maximum intensities present in the image:
%-----------------------------------------------------------------
min(P(:)), max(P(:));
r_min = double(min(P(:)));
r_max = double(max(P(:)));
disp("Max Value=")
disp(r_max);
disp("Min Value=")
disp(r_min);

%-----------------------------------------------------------------
% d. Perform contrast stretching
%-----------------------------------------------------------------
% Formula ~ s=(255*(r-rmin)/(rmax-rmin))


P2a(:,:) = imsubtract(P(:,:), r_min);
P2a(:,:) = immultiply(P2a(:,:), im2double(255 / (r_max - r_min)));
min(P2a(:)), max(P2a(:));
r_min = double(min(P2a(:)));
r_max = double(max(P2a(:)));
disp("Max Value(After Contrast Stretching)=")
disp(r_max);
disp("Min Value(After Contrast Stretching)=")
disp(r_min);
P2 = uint8(P2a); 

%-----------------------------------------------------------------
% e. redisplay image
%-----------------------------------------------------------------
figure;
imshow(P2);
title('Stretched Image');

%-----------------------------------------------------------------
% alternative imshow(can don't do uint8)
%-----------------------------------------------------------------
imshow(P2,[]); 









%% 2.2


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.2 Histogram Equalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% a. Display the image intensity histogram of P using 10 bins by
%-----------------------------------------------------------------
imhist(P,10);

%-----------------------------------------------------------------
%a.Next display a histogram with 256 bins. What are the differences?
%-----------------------------------------------------------------
imhist(P,10);

% Differences: 256 bins is more detailed than 10 bins histogram


%-----------------------------------------------------------------
% b. Next, carry out histogram equalization as follows: 
%-----------------------------------------------------------------
P3 = histeq(P,255);

%------------------------------------------------------------------------------------
% Redisplay the histograms for P3 with 10 and 256 bins. Are the histograms equalized?
% What are the similarities and differences between the latter two histograms? 
%------------------------------------------------------------------------------------
imhist(P3,10);
imhist(P3,256);

% Answer 1: Yes,both histogram is equalized but for 256 bins it does
% not equalize as much as 10 bins histogram

% Answer 2: 
% Similarities: Both histogram gray level is ranged from 0 to 255
% Differences : (1) 10 bins histogram is more equalized than 256 bins histogram
%               (2) 10 bins histogram frequency is around 1.5*10^4 but 256
%               bins histogram ranged from around 334 to 2935
%               (3) 10 bins histogram is evenly spaced out but 256 bins
%               histogram has a higher spaced out from 161 to 239


%----------------------------------------------------------------------------------------
% c. Rerun the histogram equalization on P3. Does the histogram become more uniform? Give
% suggestions as to why this occurs.
%----------------------------------------------------------------------------------------

P3 = histeq(P3,255);
imhist(P3,10);
imhist(P3,256);

% Answer : No, the histogram is the same as before.I think it is due to
%          that the bins have already been assigned to their "designated" 
%          area during the first equalization which is by cumulative
%          probability function so rerunning equalization will not have any
%          changes

% EXTRA: Checking of difference of image after histogram equalizatiom

figure;
imshow(P3);
title('After Histogram Equalization');






%% 2.3


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.3 Linear Spatial Filtering 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In linear spatial filtering, typically an input image f(x,y) is convolved with a 
% small spatial filter h(x,y) to create an output image g(x,y)
% The function h(x,y) can be considered to be the impulse response of the filter system.
% Another common name for h(x,y) is the point-spread function (PSF).



% Gaussian Filter Documentation MATLAB
% https://ww2.mathworks.cn/help/images/ref/fspecial.html#d123e14277

%----------------------------------------------------------------------------------------
% a. Generate the following filters:
%----------------------------------------------------------------------------------------
% % h(n)=......
h=@(X,Y,std) exp(-(X.^2.+Y.^2)./(std^2.*2))./(2*pi*std^2);
% dimension= ......, -2, -1, 0, 1, 2, ......
dim=5;
x=-(dim-1)/2:(dim-1)/2;y=-(dim-1)/2:(dim-1)/2;
% 
% create x,y array
[X,Y] = meshgrid(x,y);


% % (i) Y and X-dimensions are 5 and σ = 1.0
h1=h(X,Y,1);
% Normalization
h1_norm=h1./sum(h1,'all');
figure
% 指定值为 0.8 的 FaceAlpha 名称-值对组
mesh(X,Y,h1_norm,'FaceAlpha','0.8')
title('Gaussian Filter (Standard Deviation=1.0)'),xlabel('X'),ylabel('Y'),zlabel('Z');


% (ii) Y and X-dimensions are 5 and σ = 2.0
h2=h(X,Y,2);
% Normalization
h2_norm=h2./sum(h2,'all');
figure
mesh(X,Y,h2_norm,'FaceAlpha','0.8')
title('Gaussian Filter (Standard Deviation=2.0)'),xlabel('X'),ylabel('Y'),zlabel('Z');

%----------------------------------------------------------------------------------------
% b. Download the image (ntu-gn.jpgb from NTULearn and view it. Notice that this image has
% additive Gaussian noise.
%----------------------------------------------------------------------------------------
% Already done
lib_gn = imread('lib-gn.jpg');
imshow(lib_gn);

%----------------------------------------------------------------------------------------
% c. Filter the image using the linear filters that you have created above using the conv2
% function, and display the resultant images. How effective are the filters in removing noise?
% What are the trade-offs between using either of the two filters, or not filtering the image at
% all?
%----------------------------------------------------------------------------------------

filtered_lib_gn1 = uint8(conv2(lib_gn,h1));
filtered_lib_gn2 = uint8(conv2(lib_gn,h2));

imshow(filtered_lib_gn1);
title('Filtered Library GN using h(n) with std=1');
imshow(filtered_lib_gn2);
title('Filtered Library GN using h(n) with std=2');

% Answer : The filters are effective in removing Gaussian noise and the
%          higher the standard deviation the more noise is being removed
%
%          The trade-off of using the filter is the picture will become
%          more blurred because removing noise will blurred the edges also
%          but if picture is not being filtered then we will have a picture
%          that has a lot of noise which make the picture not clear also


%----------------------------------------------------------------------------------------
% d. Download the image %ntu-sp.jpg  from NTULearn and view it. Notice that this image has
% additive speckle noise. 
%----------------------------------------------------------------------------------------
lib_sp = imread('lib-sp.jpg');
imshow(lib_sp);

%----------------------------------------------------------------------------------------
% e. Repeat step (c) above. Are the filters better at handling Gaussian noise or speckle noise?
%----------------------------------------------------------------------------------------
filtered_lib_sp1 = uint8(conv2(lib_sp,h1));
filtered_lib_sp2 = uint8(conv2(lib_sp,h2));

imshow(filtered_lib_sp1);
title('Filtered Library SP using h(n) with std=1');
imshow(filtered_lib_sp2);
title('Filtered Library SP using h(n) with std=2');

% Answer: The filters are able to remove the speckle noise also but it is
%         not as effective as removing gaussian noise as there is still
%         some visible but not very clear small white dot in the pictures










%% 2.4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.4 Median Filtering
%
%
% Median filtering is a special case of order-statistic filtering. For each pixel, the set of
% intensities of neighbouring pixels (in a neighbourhood of specified size) are ordered. Median
% filtering involves replacing the target pixel with the median pixel intensity. Repeat steps (b)-(e)
% in Section 2.3, except instead of using h(x,y) and conv2, use the command medfilt2 with
% different neighbourhood sizes of 3x3 and 5x5. How does Gaussian filtering compare with
% median filtering in handling the different types of noise? What are the tradeoffs?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mfiltered_libgn1 = medfilt2(lib_gn,[3,3]);
mfiltered_libgn2 = medfilt2(lib_gn,[5,5]);
imshow(mfiltered_libgn1);
title('Filtered Library GN using Median Filter(3X3)');
imshow(mfiltered_libgn2);
title('Filtered Library GN using Median Filter(5X5)');

mfiltered_libsp1 = medfilt2(lib_sp,[3,3]);
mfiltered_libsp2 = medfilt2(lib_sp,[5,5]);
imshow(mfiltered_libsp1);
title('Filtered Library SP using Median Filter(3X3)');
imshow(mfiltered_libsp2);
title('Filtered Library SP using Median Filter(5X5)');

% Answer : Gaussian Filtering is better at removing gaussian noise compared to Median 
%          Filtering whereas Median Filtering is better at removing speckle noise 
%          compared to Gaussian Filtering and the more the neighbourhood
%          size the better the removing of noise for both noises.
%
%          Using Median Filtering, speckle noise which is the white dot in 
%          the picture can be removed efficiently and the edges can be 
%          preserved way better than using Gaussian Filtering but in 
%          trade-off which the gaussian noise is not able to be removed 
%          easily. Increasing of neighbourhood size will increase the 
%          efficiency of removing the noises in trade-off of edges will not
%          be preserved that well.








%% 2.5


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.5 Suppressing Noise Interference Patterns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% a. Download the image %pck-int.jpg  from NTULearn and display it from MATLAB. Notice the
% dominant diagonal lines destroying the quality of the image.
%----------------------------------------------------------------------------------------

pck_int=imread('pck-int.jpg');
imshow(pck_int);



% b. Obtain the Fourier transform F of the image using fft2, and subsequently compute the
% power spectrum S. Note that F should be a matrix of complex values, but S should be a
% real matrix. Display the power spectrum by
%----------------------------------------------------------------------------------------

F = fft2(pck_int);
S = abs(F);

figure;
imagesc(fftshift(S.^0.1));
colormap('default');
title('FFShifted FT Power Spectrum');
figure;
imagesc(fftshift(S.^0.1));
colormap('gray');
title('FFShifted FT Power Spectrum');


% c. Redisplay the power spectrum without fftshift. Measure the actual locations of the peaks.
% You can read the coordinates off the x and y axes, or you can choose to use the ginput
% function.
%----------------------------------------------------------------------------------------

figure;
imagesc(S.^0.1);
colormap('default');
title("FT Power Spectrum")
figure;
imagesc(S.^0.1);
colormap('gray');
title("fft (Grey)")

% Locate coordinate (x,y) using ginput() function
% https://www.mathworks.com/help/matlab/ref/ginput.html
[n,m] = ginput(1);



% d. Set to zero the 5x5 neighbourhood elements at locations corresponding to the above
% peaks in the Fourier transform F, not the power spectrum. Recompute the power
% spectrum and display it as in step (b).
%----------------------------------------------------------------------------------------
% rounded integers of peaks from c.
% Top Right FFT Power Spectrum
n=[15, 19]
m=[247,251]
%  
% Bottom left FFT Power Spectrum
n=[239,243]
m=[7,11]

F(15:19,247:251)=0;
F(239:243,7:11)=0;
S=abs(F).^2;
figure;
imagesc(fftshift(S.^0.1));
colormap('default');
title("FFShifted FT Power Spectrum(Recomputed)")
figure;
imagesc(fftshift(S.^0.1));
colormap('gray');
title("FFShifted FT Power Spectrum(Recomputed)")


% e. Compute the inverse Fourier transform using ifft2 and display the resultant image.
% Comment on the result and how this relates to step (c). Can you suggest any way to
% improve this? 
%----------------------------------------------------------------------------------------
figure;
colormap('default');
imshow(uint8(ifft2(F)))


% Answer : Most of the interference pattern are removed when we set F(n,m)=0 which n
%          and m are the coordinates of the peak.When F(n,m) is set to 0,
%          the specific sets of building block('atom') of the image that is corresponds
%          to the interference patterns is being removed and when we use
%          the inverse Fourier Transform the specific atom with 'weight'=0
%          will have no influence on the image
         

% f. Download the image `primate-caged.jpg  from NTULearn which shows a primate behind a
% fence. Can you attempt to "free" the primate by filtering out the fence? You are not likely
% to achieve a clean result but see how well you can do. 
%----------------------------------------------------------------------------------------

primate_caged = imread('primate-caged.jpg');
% Fourier transform for rgb image by right need to do transform for each color 
% respectively but since this looks like a grayscale image i just converted it
% to a gray image (tested on colormap but not working end up finding that this 
% jpg is rgb)
primate_caged = rgb2gray(primate_caged);
imshow(primate_caged);

F = fft2(primate_caged);   
S = abs(F);
imagesc(S.^0.01);
% title('Primate Caged Power Spectrum')
% colormap('default');
imagesc(fftshift(S.^0.1));
title('Primate Caged Power Spectrum(fftshift)')
colormap('default');
imagesc(fftshift(S.^0.1));
title('Primate Caged Power Spectrum(fftshift grey)')
colormap('gray');
[n,m] = ginput(1);

%top right
x1 = 6;
y1 = 247;
F(x1-2:x1+2, y1-2:y1+2) = 0;

x2 = 10;
y2 = 237;
F(x2-2:x2+2, y2-2:y2+2) = 0;
 
% %bottom left
x3 = 252;
y3 = 11;
F(x3-2:x3+2, y3-2:y3+2) = 0;

x4 = 248;
y4 = 22;
F(x4-2:x4+2, y4-2:y4+2) = 0;

S = abs(F);
imagesc(S.^0.1);
colormap('default');

ft_primatecaged = uint8(ifft2(F));
imshow(ft_primatecaged);
title('Fourier Transform Primate Caged')





%% 2.6


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.6 Undoing Perspective Distortion of Planar Surface
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% a. Download `book.jpg% from the NTULearn website as a matrix P and display the image.
% The image is a slanted view of a book of A4 dimensions, which is 210 mm x 297 mm. 
%----------------------------------------------------------------------------------------
P=imread('book.jpg');
imshow(P)


% b. The ginput function allows you to interactively click on points in the figure to obtain the
% image coordinates. Use this to find out the location of 4 corners of the book, remembering
% the order in which you measure the corners.
%----------------------------------------------------------------------------------------
[X,Y] = ginput(4);

% c. Set up the matrices required to estimate the projective transformation based on the
% equation (*) above.
%      >> u = A \ v;
% The above computes u = A-1
% v, and you can convert the projective transformation
% parameters to the normal matrix form by
%      >> U = reshape([u;1], 3, 3)';
% Write down this matrix. Verify that this is correct by transforming the original coordinates
%      >> w = U*[X'; Y'; ones(1,4)];
%      >> w = w ./ (ones(3,1) * w(3,:))
% Does the transformation give you back the 4 corners of the desired image? 
% begins Top left rotate clockwise
%----------------------------------------------------------------------------------------

X_im = [0; 210; 210; 0];
Y_im = [0; 0; 297; 297];


A = [
    [X(1), Y(1), 1, 0, 0, 0, -X_im(1) * X(1), -X_im(1) * Y(1)];
    [0, 0, 0, X(1), Y(1), 1, -Y_im(1) * X(1), -Y_im(1) * Y(1)];
    [X(2), Y(2), 1, 0, 0, 0, -X_im(2) * X(2), -X_im(2) * Y(2)];
    [0, 0, 0, X(2), Y(2), 1, -Y_im(2) * X(2), -Y_im(2) * Y(2)];
    [X(3), Y(3), 1, 0, 0, 0, -X_im(3) * X(3), -X_im(3) * Y(3)];
    [0, 0, 0, X(3), Y(3), 1, -Y_im(3) * X(3), -Y_im(3) * Y(3)];
    [X(4), Y(4), 1, 0, 0, 0, -X_im(4) * X(4), -X_im(4) * Y(4)];
    [0, 0, 0, X(4), Y(4), 1, -Y_im(4) * X(4), -Y_im(4) * Y(4)];
];


v = [X_im(1); Y_im(1); X_im(2); Y_im(2); X_im(3); Y_im(3); X_im(4); Y_im(4)];

u = A\v;

U = reshape([u;1], 3, 3)';

w = U*[X'; Y'; ones(1,4)];

w = w ./ (ones(3,1) * w(3,:));
disp(w);

% d.Warp the image via 
%----------------------------------------------------------------------------------------

T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);


% e. Display the image. Is this what you expect? Comment on the quality of the transformation
% and suggest reasons. 
%----------------------------------------------------------------------------------------
imshow(P2)
title('Frontial View of the book');

% Answer: The quality of the book picture is not as good as directly take a
%         new picture of frontial view of the book because the picture is 
%         cropped and warp into new dimension of picture which the top part
%         of the picture is more blurry than the bottom part of the pciture
%         ,the exactly same as the original picture used.


% f. In your new image, identify the big rectangular pink area, which seems to be a computer
% screen, at about middle place between "Nanyang" and “2001”. You may use any methods
% you wish.
%----------------------------------------------------------------------------------------
img = P2;
red_layer   = img(:,:,1);
green_layer = img(:,:,2);
blue_layer  = img(:,:,3);
% Define RGB code range for orange color

% Apply thresholds(Retry)
orange_pos = red_layer>=196 &red_layer<=220 & ...
             green_layer>=114&green_layer<=150 & ...
             blue_layer>=90&blue_layer<=150;

BW = orange_pos;
maskedRGBImage = img;
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

imshow(maskedRGBImage);

[r, c] = find(BW);
row1 = min(r);
row2 = max(r);
col1 = min(c);
col2 = max(c);
disp(row1);
disp(row2);
disp(col1);
disp(col2);

croppedImage = imcrop(img,[col1 row1 col2-col1 row2-row1]);
imshow(croppedImage);

% Trying to remove the blue line via changing to other color
%% 
originalImg = croppedImage;

% red_layer   = img(:,:,1);
% green_layer = img(:,:,2);
% blue_layer  = img(:,:,3);

red_layer   = originalImg(:,:,1);
green_layer = originalImg(:,:,2);
blue_layer  = originalImg(:,:,3);
% Define RGB code range for orange color

% Apply thresholds for finding blue part
blue_pos = red_layer>=30 &red_layer<=150& ...
             green_layer>=30&green_layer<=150 & ...
             blue_layer>=60&blue_layer<=255;

BW = blue_pos;

imshow(BW);

red_layer(BW)=217;
green_layer(BW)=120;
blue_layer(BW)=107;

rgbImage = cat(3, red_layer, green_layer, blue_layer);

%enhance image with a filter
nImage = rgbImage;
windowWidth = 1;
kernel = -1 * ones(windowWidth);
kernel(ceil(windowWidth/2), ceil(windowWidth/2)) = -sum(kernel(:)) - 1 + windowWidth^2 ;
kernel = kernel / sum(kernel(:)); % Normalize.
out = imfilter(nImage, kernel);
imshow(out, []);

%%%The end of experiment 1
