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

void laplace(Mat& img, Mat& imgLa)
{
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	filter2D(img, imgLa, CV_8UC3, kernel);
}

void yulin(Mat &img, Mat &cImage)
{
	Mat tImage;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	cv::morphologyEx(img, tImage, cv::MORPH_TOPHAT, element);
	threshold(tImage, cImage, 35.0, 255.0, CV_THRESH_BINARY_INV);
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
	{	cout << coefs[i] << endl;
		for(int j = 0; j < nRows; j++)
		{	
			temp = (float)src.at<uchar>(j, i);
			temp *= coefs[i];		
			src.at<uchar>(j, i) = (uchar)temp;
		}

	}

	threshold(src, src, 60.0, 255.0, CV_THRESH_BINARY_INV);
}


int main(int argc, char* argv[])
{
	//Mat img = imread("/home/xuesong/RailInspection/defectsImages/test/1.jpg");
	//img = img(Rect(250, 0, 180, 480));
	Mat img = imread("/home/xuesong/RailInspection/defectsImages/yulinshang/y2.png",0);
		cout << img.cols << endl;

	if (img.empty())
		return -1;
	if (img.channels() != 1)
	{
		cvtColor(img, img, CV_BGR2GRAY);
	}
	imshow("raw", img);


	Mat imgLog;
	log_normalization(img, imgLog);
	imshow("log",imgLog);
	
	Mat imgHist;
	equalizeHist(img, imgHist);
	imshow("hist",imgHist);

	Mat imgLa;
	laplace(img, imgLa);
	imshow("laplace", imgLa);

	Mat imgTop;
	yulin(img, imgTop);
	imshow("yulin", imgTop);
	
	double t0 = (double)getTickCount();
	Mat src;
	diaokuai(img, src);
	imshow("diaokuai", src);
	
	t0 = (double)getTickCount() - t0;
	cout << " time:\n " << t0 * 1000 / getTickFrequency() << " ms" << endl;
	waitKey();	

	return 0;
}
