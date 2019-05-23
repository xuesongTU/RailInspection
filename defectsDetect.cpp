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
const int G = 100; // compensation coef
const int THR_Y = 40; // threshold to judge if there is yulinshang
const int THR_D = 25; // threshold to judge if there is diaoluai

void log_normalization(Mat& Img, Mat& imgL)
{
	if (Img.depth() == CV_8U)
	{
		Img.convertTo(imgL, CV_32F);
	}

	imgL = (imgL + 1)*0.5;
	Mat logImg;
	log(imgL, logImg);

	Mat tmp_m, tmp_std;
	double meanV, stdV;
	meanStdDev(logImg, tmp_m, tmp_std);
	meanV = tmp_m.at<double>(0, 0);
	stdV = tmp_std.at<double>(0, 0);

	logImg = (logImg - meanV) / stdV;

	normalize(logImg, imgL, 1.0, 0.0, NORM_MINMAX);
}


void yulin(Mat &img, Mat &cImage)
{
	Mat tImage;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	cv::morphologyEx(img, tImage, cv::MORPH_TOPHAT, element);
	threshold(tImage, cImage, 40.0, 255.0, CV_THRESH_BINARY_INV);
}


void diaokuai(Mat& img, Mat& src)
{	
	src = img.clone();
	int nCols = src.cols;
	int nRows = src.rows;
	vector<float> coefs;
	for(int i = 0; i < nCols; i++)
	{	
		float mean = 0.;
		for(int j = 0; j < nRows; j++)
		{
			mean += src.at<uchar>(j, i);
		}
		mean /= nRows;
		coefs.push_back(G / mean);
	}
	
	double temp;
	for(int i = 0; i < nCols; i++)
	{	
		for(int j = 0; j < nRows; j++)
		{	
			temp = (float)src.at<uchar>(j, i);
			temp *= coefs[i];		
			src.at<uchar>(j, i) = (uchar)temp;
		}

	}

	threshold(src, src, 60.0, 255.0, CV_THRESH_BINARY);
}


bool isDefect(Mat &img, int thresh)
{	
	bool flag = false;
	int len, num;
	for(int i = 0; i < img.cols; i++)
	{	
		num = 0;
		len = 0;
		for(int j = 0; j < img.rows; j++)
		{
			if(img.at<uchar>(j, i) == 0)
			{
				num ++;
				if(num > len)
					len = num;
			}
			else
			{
				num = 0;
			}
		}
		if(num > thresh)
		{
			flag = true;
			break;
		}
	}

	for(int i = 0; i < img.rows; i++)
	{	
		num = 0;
		len = 0;
		for(int j = 0; j < img.cols; j++)
		{
			if(img.at<uchar>(i,j) == 0)
			{
				num ++;
				if(num > len)
					len = num;
			}
			else
			{
				num = 0;
			}
		}
		if(num > thresh)
		{
			flag = true;
			break;
		}
	}

	return flag;
}

int main(int argc, char* argv[])
{
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

	Mat imgY;
	yulin(img, imgY);
	imshow("yulin", imgY);
	if(isDefect(imgY, THR_Y))
		cout << "yulin" << endl;
	

	Mat imgD;
	diaokuai(img, imgD);
	imshow("diaokuai", imgD);
	if(isDefect(imgD, THR_D))
	cout << "diaokuai" << endl;
	
	t0 = (double)getTickCount() - t0;
	cout << " time:\n " << t0 * 1000 / getTickFrequency() << " ms" << endl;
	waitKey();	

	return 0;
}
