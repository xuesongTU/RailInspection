#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <vector>  
#include <algorithm>

using namespace std;
using namespace cv;

//adaptive Median Filter

uchar adaptiveProcess(const Mat &im, int row,int col,int kernelSize,int maxSize)
{
    vector<uchar> pixels;
    for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
        for (int b = -kernelSize / 2; b <= kernelSize / 2; b++)
        {
            pixels.push_back(im.at<uchar>(row + a, col + b));
        }
    sort(pixels.begin(), pixels.end());
    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = im.at<uchar>(row, col);
    if (med > min && med < max)
    {
        // to B
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveProcess(im, row, col, kernelSize, maxSize);
 //增大窗口尺寸，继续A过程。
        else
            return med;
    }
}


void log_normalization(Mat& Img)
{
	Img = (Img + 1)*0.5;
	Mat logImg;
	cout << logImg << endl;
	log(Img, logImg);

	Mat tmp_m, tmp_std;
	double meanV, stdV;
	meanStdDev(logImg, tmp_m, tmp_std);
	meanV = tmp_m.at<double>(0, 0);
	stdV = tmp_std.at<double>(0, 0);

	logImg = (logImg - meanV) / stdV;
	normalize(logImg, Img, 1.0, 0.0, NORM_MINMAX);
}


void diaokuai(Mat &img, Mat &src, double thresh)
{
	threshold(img, src, thresh, 255.0, CV_THRESH_BINARY_INV);
}

void yulin(Mat &img, Mat &cImage)
{
	Mat tImage;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	cv::morphologyEx(img, tImage, cv::MORPH_TOPHAT, element);
	threshold(tImage, cImage, 35.0, 255.0, CV_THRESH_BINARY_INV);
}

void adaptiveMedianFilter(Mat &img, int nRow, int nCol, int winSize)
{
	//扩展图像的边界
	//copyMakeBorder(img, img, winSize / 2, winSize / 2, winSize / 2, winSize / 2, BorderTypes::BORDER_REFLECT);
	
	for(int i = winSize / 2; i < nRow - winSize / 2; i++)
	{
		for(int j = winSize / 2; j < nCol - winSize / 2; j++)
		{
			vector<uchar> pixels;
			for (int a = -winSize / 2; a <= winSize / 2; a++)
			{
				for (int b = -winSize / 2; b <= winSize / 2; b++)
				{
				    pixels.push_back(img.at<uchar>(i + a, j + b));
				}
			}
			sort(pixels.begin(), pixels.end());
			if(img.at<uchar>(i, j) == pixels.back());
				img.at<uchar>(i,j) = pixels[winSize*winSize / 2];
		}	
	}
}

int main(int argc, char* argv[]){
	//Mat img = imread("/home/xuesong/RailInspection/defectsImages/yulinshang/g1.png");
	//Mat img = imread("/home/xuesong/RailInspection/defectsImages/diaokuai/d5.png");
	//resize(img, img, Size(200, 1000), 0, 0, CV_INTER_LINEAR);

	Mat img = imread("/home/xuesong/RailInspection/defectsImages/test/14.jpg");
	img = img(Rect(250, 0, 180, 480));

	
	double t0 = (double)getTickCount();

	if (img.empty())
		return -1;
	if (img.channels() != 1)
	{
		cvtColor(img, img, CV_BGR2GRAY);
	}

	imshow("raw", img);
	if (img.depth() == CV_8U)
	{
		img.convertTo(img, CV_32F);
	}
	

	
	
	log_normalization(img);
	imshow("normalization", img);
	
	Mat dImg;
	diaokuai(img, dImg, 50.0);
	imshow("diaokuai", dImg);


	Mat aImage;	
	adaptiveMedianFilter(img, img.cols, img.rows, 5);
	imshow("adp", img);


	Mat bImage;	
	medianBlur(img, bImage, 5);
	imshow("medianblur", bImage);
	


/*
	int minSize = 5; // 滤波器窗口的起始尺寸
    int maxSize = 7; // 滤波器窗口的最大尺寸
    Mat im1;
    // 扩展图像的边界
    copyMakeBorder(img, im1, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BorderTypes::BORDER_REFLECT);
    // 图像循环
    for (int j = maxSize / 2; j < im1.rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < im1.cols * im1.channels() - maxSize / 2; i++)
        {
            im1.at<uchar>(j, i) = adaptiveProcess(im1, j, i, minSize, maxSize);
        }
    }
	
	imshow("median_filter", im1);
*/	
	//double t0 = (double)getTickCount();
	//top-hat operation
	//cv::Mat element(4, 4, CV_8U, cv::Scalar(1));
	
	Mat cImage;
	yulin(img, cImage);
	imshow("Binary", cImage);

	t0 = (double)getTickCount() - t0;
	cout << " time:\n " << t0 * 1000 / getTickFrequency() << " ms" << endl;
	imshow("tophat", img);
	waitKey();


	return 0;
}
