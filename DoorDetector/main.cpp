// DoorDetector
// This program exists to scan images through
// a Haar cascade classifier to determine if
// they are pictures of container doors or
// container fronts.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
using namespace cv;

//Function prototypes
Mat MakeHistogram(Mat& image);
Mat CascadeDetector(Mat& image, const char XML[100]);
Mat IlluminateImage(Mat& image);
Mat Sharpen(Mat& inputImage);
Mat CloseImage(Mat& inputImage);
Mat OpenImage(Mat& inputImage);
void Usage(void);
void SaveFile(Mat& image, const char* path, const string& description);



int main(int argc, char * const * argv)
{
    // These lines are part of getopt.
    // They just make sure the passed arguments are valid.
    int ch = 0;
    
    char *xml_filename = nullptr;
    
    while ((ch = getopt(argc, argv, "dx:")) != -1)
    {
        switch (ch) {
            case 'x':
                xml_filename = optarg;
                break;
            default:
                Usage();
        }
    }
    argc -= optind;
    argv += optind;
    
    if (!xml_filename || argc == 0)
    {
        Usage();
    }
    //End getopt.
    
    
    //Make empty image structures
    
    cv::Mat inputImage;
    cv::Mat brightenedImage;
    cv::Mat detectedImage;
    
    //Enumerate through the image arguments.
    
    for(int input=0; input<(argc); input++)
    {
        inputImage = cv::imread(argv[input]);    //Read the image
        
        if(inputImage.empty())                   //Prevent running on an empty image
        {
            return true;
        }
        
        //First the image is passed to the IlluminateImage function.
        //If the image needs to be brightened, it will do so and then
        //it will pass it to the detector function.
        //If not, then it will not alter the image, and still pass it
        //to the detector function.
        
        brightenedImage = IlluminateImage(inputImage);
        detectedImage = CascadeDetector(brightenedImage, xml_filename);
        
    }
    
    
    return 0;
}

//------------------------------------------------------------------------------------
//Functions
//------------------------------------------------------------------------------------

//Each function returns a Mat image. This is for testing and debugging, where it is
//useful to have multiple images stored and displayed.

//As of 2015-07-01 when I am sending this, not all functions are used.
//There exist other functions to apply image processing, but they have mixed results.
//These other functions can be implemented quickly.
//The functions being used right now are:
//CascadeDetector
//IlluminateImage

//------------------------------------------------------------------------------------
//This function runs the cascade detector on an image and prints the result.
//It has sections that will draw rectangles or put text on the photos.
//These are meant for debugging.
Mat CascadeDetector(Mat& image, const char XML[100])
{
    Mat detectedImage;
    image.copyTo(detectedImage);
    CascadeClassifier cascade;          //Define the detector
    if(cascade.load(XML))           //Only go if the detector is loaded
    {
        
        std::vector<Rect> supports;     //Array of rectangles for the bounding boxes
        cascade.detectMultiScale(detectedImage, //image to be scanned
                                 supports, //array to store results in
                                 1.1,   //scaling factor
                                 40,    //minimum neighbors
                                 0|CV_HAAR_SCALE_IMAGE, //flag to prevent the detector from detecting twice
                                 Size(30,30),   //minimum size
                                 Size(100,100)); //maximum size
        
        //Runs the passed image through the detector. The rest of the parameters define how many boxes can be detected and of what sizes. The detected objects are stored as rectangles in the vector supports.
        
        //These statements identify whether the photo is a front or a door. The commented lines will put yellow text
        //onto the image directly to identify them during testing.
        
        if(supports.size() <= 2)
        {
            //putText(detectedImage, "Front", cvPoint(200,300), FONT_HERSHEY_COMPLEX, 5.0, cvScalar(0,255,255), 10, 8);
            printf("\rFront\n");
        }
        if(supports.size() >= 5)
        {
            //putText(detectedImage, "Door", cvPoint(200,300), FONT_HERSHEY_COMPLEX, 5.0, cvScalar(0,255,255), 10, 8);
            printf("\rDoor\n");
        }
        if(supports.size() == 3 || supports.size() == 4)
        {
            //putText(detectedImage, "Probably Door", cvPoint(0,300), FONT_HERSHEY_COMPLEX, 3.0, cvScalar(0,255,255), 10, 8);
            printf("\rProbably Door\n");
        }
        
        //printf("%lu\n",supports.size()); //prints how many objects were detected for debugging
        //Uncomment this for loop to actually draw the yellow boxes on the image for testing.
        /*for(auto iter = supports.begin(); iter != supports.end(); ++iter)
         {
         //Draw the bounding boxes to check
         rectangle(detectedImage,*iter,Scalar(0,255,255),1,8,0);
         }*/
    }
    return detectedImage;
}

//------------------------------------------------------------------------------------
//This function determines if a photo was taken at night, and brightens it
//by converting the photo to HLS format and altering its channels.

Mat IlluminateImage(Mat& image)
{
    int pixelNumbers = 0;
    Mat brightenedImage;
    image.copyTo(brightenedImage);
    for(int j=0; j<brightenedImage.rows;j++)
    {
        for(int i=0; i<brightenedImage.cols;i++)
        {
            if(brightenedImage.at<Vec3b>(j,i)[0] >= 128)
            {
                pixelNumbers++;
                //Photos taken at night don't have large blue values
                //Night photos are largely red
                //We count how many pixels have large blue values to determine if it is a night-time photo or not.
            }
        }
    }
    if(pixelNumbers <= 1000)
    {
        cvtColor(brightenedImage, brightenedImage, CV_BGR2HLS);
        for(int j=0; j<brightenedImage.rows;j++)
        {
            for(int i=0; i<brightenedImage.cols;i++)
            {
                //This line increases the Hue channel, and makes the photos more accurately approximate daytime.
                brightenedImage.at<Vec3b>(j,i)[0]+= 15;
            }
        }
        
        //The colorspace must be converted back to BGR for everything else to work correctly.
        cvtColor(brightenedImage, brightenedImage, CV_HLS2BGR);
        
    }
    return brightenedImage;
}



//------------------------------------------------------------------------------------
//Currently unused functions
//------------------------------------------------------------------------------------



//------------------------------------------------------------------------------------
//This function will save the images as new files.
//This is only useful for analysis.

void SaveFile(Mat& image, const char* path, const string& description)
{
    char newFileName[1024];
    snprintf(newFileName, 1024, "%s.%s.jpg", path, description.c_str());
    cv::imwrite(newFileName, image);
}

//------------------------------------------------------------------------------------
//This function creates an RGB color histogram of an image.
//Hypothetically it will create a three-part histogram no matter
//the image format. But the lines are in red, green, and blue.

Mat MakeHistogram(Mat& Image)
{
    vector<Mat> bgr_planes;
    split(Image, bgr_planes);
    /// Establish the number of bins
    int histSize = 256;
    
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    
    bool uniform = true; bool accumulate = false;
    
    Mat b_hist, g_hist, r_hist;
    
    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    
    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
    }
    
    /// Display
    //namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    //imshow("calcHist Demo", histImage );
    return histImage;
}

//------------------------------------------------------------------------------------
//This function sharpens edges in an image by subtracting the
//second derivative of pixel intensity.
//This clears up blurriness and brings out details in the image.
Mat Sharpen(Mat& inputImage)
{
    cv::Mat laplacian;
    cv::Mat inputImage_32bit;
    
    inputImage.convertTo(inputImage_32bit, CV_32F);
    Laplacian(inputImage, laplacian, CV_32F, 3);
    Mat sharp_image_32bit = inputImage_32bit - (0.3)*(laplacian);
    sharp_image_32bit.convertTo(sharp_image_32bit, CV_8U);
    return sharp_image_32bit;
}

//-------------------------------------------------------------------------------------
//This function performs a binary opening operation.
//This has the effect of eliminating noise and small features
//while preserving approximate size of the larger features.

Mat OpenImage(Mat& inputImage)
{
    cv::Mat openedImage;
    cv::Mat five_by_five_element(5,5, CV_8U, Scalar(1));
    
    morphologyEx(inputImage, openedImage, MORPH_OPEN, five_by_five_element);
    return openedImage;
}

//--------------------------------------------------------------------------------------
//This function performs a binary closing operation.
//This has the effect of joining close objects together
//and filling in holes. This distorts the shape of the objects
//while preserving size.

Mat CloseImage(Mat& inputImage)
{
    cv::Mat closedImage;
    cv::Mat five_by_five_element(5,5,CV_8U, Scalar(1));
    
    morphologyEx(inputImage, closedImage, MORPH_CLOSE, five_by_five_element);
    return closedImage;
}
//--------------------------------------------------------------------------------------
//This is a thing for getopt. It gives instructions for how to use the program.

void Usage()
{
    printf("Processing Tests -x <training file> <filename> [...]\n");
    printf("   -x xml training file\n");
    exit(-1);
}
