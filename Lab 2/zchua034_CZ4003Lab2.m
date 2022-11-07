% CZ4003 Lab 2
%% 3.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.1 Edge Detection 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----------------------------------------------------------------
% a. Download 'macritchie.jpg' from edveNTUre and convert the image to grayscale. 
% Display the image. 
%-----------------------------------------------------------------
P=imread("macritchie.jpg");
macritchie=rgb2gray(P);
imshow(macritchie);

%-----------------------------------------------------------------
% b. Create 3x3 horizontal and vertical Sobel masks and filter the image using conv2. 
% Display the edge-filtered images. What happens to edges which are not strictly 
% vertical nor horizontal, i.e. diagonal?
%-----------------------------------------------------------------
sobel_h = [
    -1 -2 -1; 
     0 0 0; 
     1 2 1;
];

sobel_v = [
    -1 0 1; 
    -2 0 2; 
    -1 0 1;
];

% convolute the sobel filter on the image, with vertical, horizontal and
% both vertical and horizontal filter
macritchie_sobel_h = conv2(macritchie, sobel_h);
macritchie_sobel_v = conv2(macritchie, sobel_v);
macritchie_sobel_all = conv2(macritchie_sobel_v,sobel_h);

figure; 
imshow(uint8(macritchie_sobel_h));
title("Sobel Filter Horizontal");
figure; 
imshow(uint8(macritchie_sobel_v));
title("Sobel Filter Vertical");
figure; 
imshow(uint8(macritchie_sobel_all));
title("Sobel Filter All");



% Answer: If we are using horizontal sobel filter, the vertical edges will be filtered
%         out and if we are using vertical sobel filter, the horizontal edges will be 
%         filtered out. For both cases, the edges which are not strictly horizontal nor 
%         vertical will become fainter instead of filtered out like the strictly 
%         horizontal/vertical edges.



%-----------------------------------------------------------------
% c. Generate a combined edge image by squaring (i.e. .^2) the horizontal and 
% vertical edge images and adding the squared images. Suggest a reason why a 
% squaring operation is carried out. 
%-----------------------------------------------------------------

combinedEdge_img=macritchie_sobel_v.^2+macritchie_sobel_v.^2;
figure;
imshow(uint8(combinedEdge_img));
title("Combined Edge Image");
figure;
imshow(uint8(sqrt(combinedEdge_img)));
title("Squared Combined Edge Image");


% Reason : After the Sobel Filters have been applied, the gradient of each 
%          pixel can be negative and what we need is the resultant magnitude 
%          of the edges and the positive and negative is just an indication 
%          of direction so squaring operation will help us to obtain the 
%          resultant magnitude of the gradient. 



%-----------------------------------------------------------------
% d. Threshold the edge image E at value t by 
%       >> Et = E>t; 
% This creates a binary image. Try different threshold values and display the binary 
% edge images. What are the advantages and disadvantages of using different 
% thresholds? 
%-----------------------------------------------------------------


Et = combinedEdge_img > 100;
figure; 
imshow(Et);
title("Combined Edge Image Threshold>100");
Et = combinedEdge_img > 1000;
figure; 
imshow(Et);
title("Combined Edge Image Threshold>1000");
Et = combinedEdge_img > 5000;
figure; 
imshow(Et);
title("Combined Edge Image Threshold>5000");
Et = combinedEdge_img > 10000;
figure; 
imshow(Et);
title("Combined Edge Image Threshold>10000");
Et = combinedEdge_img > 50000;
figure; 
imshow(Et);
title("Combined Edge Image Threshold>50000");
Et = combinedEdge_img > 100000;
figure; 
imshow(Et);
title("Combined Edge Image Threshold>100000");


% Answer       :
% Advantages of high threshold:
% (1) The higher accuracy of getting the outline of the object in image.
% (2) Lesser noises in the edges detected.
%
% Disadvantages of high threshold:
% (1) The image will have lesser edge compared to the lower threshold.
%



%-----------------------------------------------------------------
% e. Recompute the edge image using the more advanced Canny edge detection 
% algorithm with tl=0.04, th=0.1, sigma=1.0 
%          >> E = edge(I,%canny ,[tl th],sigma); 
% This generates a binary image without the need for thresholding. 
%
% (i) Try different values of sigma ranging from 1.0 to 5.0 and determine the 
% effect on the edge images. What do you see and can you give an 
% explanation for why this occurs? Discuss how different sigma are 
% suitable for 
%   (a) noisy edgel removal, and 
%   (b) location accuracy of edgels. 
% 
% (ii) Try raising and lowering the value of tl. What does this do? How does 
% this relate to your knowledge of the Canny algorithm? 
%
%-----------------------------------------------------------------

% (I)
tl=0.04;    % Low threshold
th=0.1;     % High threshold
sigma=1.0;  % Standard deviation of the Gaussian filter

E = edge(macritchie, 'canny', [tl th], sigma); % apply Canny Edge detection
figure; 
imshow(E);
title("Canny Edge Detector Used");

%  trying sigma 1 to 5
s=5;
for sigma = 1:s
    canny_i = edge(macritchie, 'canny', [tl th], sigma);
    figure; 
    imshow(canny_i);
    title("Canny Sigma = "+sigma);
end




% Answer: Lower sigma will give better location accuracy of edges but will have more noises.
%         As the sigma increases, although the noises is being removed but
%         more details of the edges is also being removed and the location
%         accuracy will drop.





% (ii)
valuesList = {0.01, 0.04, 0.08};
%***************************************
% loop to generate image after applying Canny Edge at different tl
% values in the valuesList
for tl = 1:length(valuesList)
    sigma = 1.0;
    cannyDiffTL = edge(macritchie, 'canny', [valuesList{tl} th], sigma);
    figure;
    imshow(cannyDiffTL);
    title("Canny Threshold = "+valuesList{tl});
end

% Answer: When the tl is small, there will be more edges and noises. As the 
%         tl increases, the noises will be lesser but the edges will also 
%         become less detailed.
% 
%         Canny algorithm uses hysteresis thresholding and the tl is the lower 
%         bound threshold value, any value lesser than the tl will be set to 0 
%         so that weak edges or noises can be removed.


%% 3.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.2 Line Finding using Hough Transform 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% In the section, the goal is to extract the long edge of the path in the `macritchie.jpg’ 
% image as a consistent line using the Hough transform. 

%-----------------------------------------------------------------
% a) Reuse the edge image computed via the Canny algorithm with sigma=1.0.
%-----------------------------------------------------------------

tl=0.04;    % Low threshold
th=0.1;     % High threshold
sigma=1.0;  % Standard deviation of the Gaussian filter

P_macritchie=imread("macritchie.jpg");
macritchie=rgb2gray(P_macritchie);
E = edge(macritchie, 'canny', [tl th], sigma); % apply Canny Edge detection
figure;
imshow(E);
title("Edge");


%-----------------------------------------------------------------
% b) As there is no function available to compute the Hough transform in MATLAB, we 
% will use the Radon transform, which for binary images is equivalent to the Hough 
% transform. Read the help manual on Radon transform, and explain why the 
% transforms are equivalent in this case. When are they different?
% 
%       >> [H, xp] = radon(E); 
% 
% Display H as an image. The Hough transform will have horizontal bins of angles 
% corresponding to 0-179 degrees, and vertical bins of radial distance in pixels as 
% captured in xp. The transform is taken with respect to a Cartesian coordinate 
% system where the origin is located at the centre of the image, and the x-axis 
% pointing right and the y-axis pointing up. 
%-----------------------------------------------------------------


[H, xp] = radon(E);
figure; 
imagesc(uint8(H)); % to visualise the waveforms better
xlabel('\theta (degrees)');
ylabel('x''');
colormap(gca, hot), colorbar;
title("Radon");
% figure;
% imshow(H);
% title("Radon Transform Image");





% Answer:
% 
% The Hough transform and the Radon transform are indeed very similar to each 
% other and their relation can be loosely defined as the former being a discretized 
% form of the latter.
% 
% The Radon transform is a mathematical integral transform, defined for continuous 
% functions on R(n) on hyperplanes in R(n). The Hough transform, on the other hand, 
% is inherently a discrete algorithm that detects lines (extendable to other shapes) 
% in an image by polling and binning (or voting).


%-----------------------------------------------------------------
% c) Find the location of the maximum pixel intensity in the Hough image in the form 
% of [theta, radius]. These are the parameters corresponding to the line in the 
% image with the strongest edge support.
%-----------------------------------------------------------------

maxH = max(H(:));
[radius, theta] = find(H == maxH);
radius = xp(radius); % obtain the radial coordinate for the maximum intensity

% apply an offset of -1 to θ as the x-axis of θ starting from 0. We
% also apply an offset using the y-axis range, xp on the radius. This allows us to obtain the 
% correct values of [θ, radius] for maxH.
theta=theta-1;
disp(radius);
disp(theta);


%-----------------------------------------------------------------
% d) Derive the equations to convert the [theta, radius] line representation to the 
% normal line equation form Ax + By = C in image coordinates. Show that A and B 
% can be obtained via 
% 
%       >> [A, B] = pol2cart(theta*pi/180, radius); 
%       >> B = -B; 
% 
% B needs to be negated because the y-axis is pointing downwards for image 
% coordinates.
%
% Find C. Reminder: the Hough transform is done with respect to an origin at the 
% centre of the image, and you will need to convert back to image coordinates 
% where the origin is in the top-left corner of the image.
%-----------------------------------------------------------------
[A,B] = pol2cart(theta * pi/180, radius);
B = -B;
% A = 18.39, B = 73.74

% Find center coordinate of the image
[numOfRows, numOfCols] = size(macritchie);
x_center = numOfCols / 2;
y_center = numOfRows / 2;
% x_center, y_center = 179, 145

% Obtain the C value from Ax + By = C
C = A*(A+x_center) + B*(B+y_center);
% C = 1.976e+04

disp(C);

%-----------------------------------------------------------------
% e) Based on the equation of the line Ax+By = C that you obtained, compute yl and 
% yr values for corresponding xl = 0 and xr = width of image - 1. 
%-----------------------------------------------------------------

xl = 0; 
xr = numOfCols - 1;

% Equation to find y value: y = (C - Ax)/B
yl = (C - A*xl)/B;
yr = (C - A*xr)/B;

disp(yl);
disp(yr);


%-----------------------------------------------------------------
% f) Display the original ‘macritchie.jpg’ image. Superimpose your estimated line by 
% 
%       >> line([xl xr], [yl yr]); 
% 
% Does the line match up with the edge of the running path? What are, if any, 
% sources of errors? Can you suggest ways of improving the estimation? 
%-----------------------------------------------------------------

figure; 
imshow(macritchie);
title("Result of Hough Transform Edge Detection");
path = line([xl xr], [yl yr]);
path.Color = "red";




% Answer:
% 
% From the result, we can see that the estimated edge is aligned with the running track 
% but if we see it closely we can found out that the edge is almost but not perfectly 
% aligned to the running track.
% 
% These are the several possibilities that I can think of:
% 1.	The line of the running track is not necessarily straight, maybe there 
% will be some curve in between 
% 2.	There can be small precision error conversion from Radon transform 
% parameters to the coordinate in image space.
% 3.	There might be some noises that will affect to get the maximum intensity pixel.
% 
% 
% Suggestion:
% 1.	We can try to use non-linear function when the running track is not totally straight.
% 2.	Try using a larger sigma value to reduce the noises on the picture


%% 3.3 

% ***PLEASE RUN THE FUNCTION SECTION BEFORE RUNNING THIS SECTION IF YOU ARE
% RUNNING IN SECTION

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.3 3D Stereo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is a fairly substantial section as you will need to write a MATLAB function script 
% to compute disparity images (or maps) for pairs of rectified stereo images Pl and Pr. 
% The disparity map is inversely proportional to the depth map which gives the distance 
% of various points in the scene from the camera. 

%-----------------------------------------------------------------
% a) Write the disparity map algorithm as a MATLAB function script which takes two 
% arguments of left and right images, and 2 arguments specifying the template 
% dimensions. It should return the disparity map. Try and minimize the use of for
% loops, instead relying on the vector / matrix processing functions. 
%-----------------------------------------------------------------

%***FUNCTION NEED TO BE AT THE BELOW ,PLEASE LOOK AT THE END OF THE CODE FILE



%-----------------------------------------------------------------
% b) Download the synthetic stereo pair images of ‘corridorl.jpg’ and ‘corridorr.jpg’, 
% converting both to grayscale
%-----------------------------------------------------------------

P_corrleft = imread('corridorl.jpg');
P_corrright = imread('corridorr.jpg');

% Convert to grayscale
P_corrleft_gray = rgb2gray(P_corrleft);
P_corrright_gray = rgb2gray(P_corrright);

figure; 
imshow(P_corrleft_gray);
title("Corridor Left");

figure;
imshow(P_corrright_gray);
title("Corridor Right");


%-----------------------------------------------------------------
% c) Run your algorithm on the two images to obtain a disparity map D, and see the 
% results via 
%
%       >> imshow(-D,[-15 15]); 
%
% The results should show the nearer points as bright and the further points as 
% dark. The expected quality of the image should be similar to `corridor_disp.jpg’ 
% which you can view for reference. 
% 
% Comment on how the quality of the disparities computed varies with the 
% corresponding local image structure. 
%-----------------------------------------------------------------

D = disparityMap(P_corrleft_gray, P_corrright_gray, 11, 11);

figure;
imshow(D,[-15 15]);
title("Result: Disparity Map D of Corridor");

% ****************
% COMMENT:
% ****************


%-----------------------------------------------------------------
% d) Rerun your algorithm on the real images of ‘triclops-i2l.jpg’ and triclops-i2r.jpg’. 
% Again you may refer to ‘triclops-id.jpg’ for expected quality. How does the image 
% structure of the stereo images affect the accuracy of the estimated disparities? 
%-----------------------------------------------------------------

P_triclops_left = imread('triclopsi2l.jpg');
P_triclops_right = imread('triclopsi2r.jpg');

% Convert to grayscale
P_triclops_left_gray = rgb2gray(P_triclops_left);
P_triclops_right_gray = rgb2gray(P_triclops_right);

D_triclops = disparityMap(P_triclops_left_gray, P_triclops_right_gray, 11, 11);
figure; 
imshow(D_triclops,[-15 15]);
title("Result: Disparity Map D of Triclops");



% ****************
% ANSWER:
% ****************


%% OPTIONAL

% Attempting the optional section will earn you extra credit. However, this is not 
% necessary to achieve a decent coursework grade. 
% 
% You will need to implement the algorithm in the CVPR 2006 paper entiled “Beyond 
% Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene 
% Categories”. You will need to use the benchmark Caltech-101 dataset and compare 
% the classification results of Spatial Pyramid Matching (SPM) and the bag-of-words 
% (BoW) method as in Table 2 of the paper by following the experimental setting in 
% Section 5.2 of the paper.






%% Function for 3.3a

% Disparity Map Algorithm
%
% for each pixel in Pl, 
%     i. Extract a template comprising the 11x11 neighbourhood region around 
%     that pixel. 
% 
%     ii. Using the template, carry out SSD matching in Pr, but only along the 
%     same scanline. The disparity is given by 
% 
%           d(x1 , y1) = x1 - xrˆ
% 
%     where x1 and y1 are the relevant pixel coordinates in Pl, 
%     and r xˆ is the SSD matched pixel’s x-coordinate in Pr. 
% 
%     You should also constrain your 
%     horizontal search to small values of disparity (<15). 
%     Noted that you may use conv2, ones and rot90 functions (may be more 
%     than once) to compute the SSD matching between the template and the 
%     input image. Refer to the equation in section 2.3 for help. 
% 
%     iii. Input the disparity into the disparity map with the same Pl pixel 
%     coordinates. 

function [result] = disparityMap(imageLeft, imageRight, noOfrow, noOfcol)
    % Convert image matrix to double
    imageLeft = im2double(imageLeft);
    imageRight = im2double(imageRight);
    
    % Obtain the dimension of left image
    [height, width] = size(imageLeft); 
 
    % Get the disparity range and the template dimensions
    rowLength = floor(noOfrow/2);
    colLength = floor(noOfcol/2);
    dispRange = 15;
    
    % Initialise matrix of 0s
    result = zeros(size(imageLeft));

    % Outer loop - each pixel along the matrix column
    for i = 1:height
        % variables to keep the range in check
        minRow = max(1, i - rowLength);
        maxRow = min(height, i + rowLength);

        % loop for each pixel along the matrix row
        for j = 1:width
            % variables to keep the range in check
            minCol = max(1, j - colLength);
            maxCol = min(width, j + colLength);
            
            % variables to keep the horizontal search range in check
            minDisp = max(-dispRange, 1 - minCol); 
            maxDisp = min(dispRange, width - maxCol);
            
            % Obtain the template from the right image
            dispTemplate = imageRight(minRow:maxRow, minCol:maxCol);

            % Initialise the variables for SSD comparison
            minSSD = inf;
            leastDifference = 0;
            
            % Inner loop - to do the searching in the search range
            for k = minDisp:maxDisp
                % Get the difference between left and right images
                newMinCol = minCol + k;
                newMaxCol = maxCol + k;
                block = imageLeft(minRow:maxRow, newMinCol:newMaxCol);
                
                % Perform SSD
                squaredDifference = (dispTemplate - block).^2;
                ssd = sum(squaredDifference(:));
                
                % Get the lowest SSD
                if ssd < minSSD
                    minSSD = ssd;
                    leastDifference = k - minDisp + 1;
                end
            end
            
            % Return the SSD result
            result(i, j) = leastDifference + minDisp - 1;
        end
    end
end

