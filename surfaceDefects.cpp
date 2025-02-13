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

int bandwidth = 3;
float gaussianIndex = 1.0 / sqrtf(6.18);
int iternum = 10;
int clusternum = 20;
float thr_segment = 0.02; //This parameter is very important 


void mean_shift(Mat& imgMat, Mat& clustersMat, Mat& weightsMat, Mat& labelMat)
{
	int imgrows = imgMat.rows;
	int imgcols = imgMat.cols;
	int repoptsnum = imgrows / 4;
	Mat roughClusterMat = Mat::zeros(repoptsnum, imgcols, CV_32F);
	RNG rng;
	int ii = 0, jj = 0, choseRep;
	Mat repMat, extendMat;
	Mat upMat, downMat, outMat;

	while (ii < repoptsnum)
	{
		choseRep = int(rng.uniform(0, imgrows));
		repMat = imgMat.row(choseRep).clone();
		while (jj++ < iternum){
			extendMat = repeat(repMat, imgrows, 1);
			extendMat = (extendMat - imgMat) / bandwidth;
			extendMat = (extendMat.mul(extendMat)) * (-0.5);
			exp(extendMat, extendMat);
			extendMat = extendMat * gaussianIndex;

			reduce(extendMat, downMat, 0, CV_REDUCE_SUM);
			reduce(extendMat.mul(imgMat), upMat, 0, CV_REDUCE_SUM);

			repMat = upMat / downMat;
		}
		repMat.copyTo(roughClusterMat.row(ii));
		ii++;
	}
	cv::sort(roughClusterMat, roughClusterMat, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	
	roughClusterMat.row(0).copyTo(clustersMat);
	int intergap = repoptsnum / (clusternum - 1);
	for (ii = 1; ii < clusternum-1; ii++)
	{
		clustersMat.push_back(roughClusterMat.row(ii*intergap));
	}
	clustersMat.push_back(roughClusterMat.row(repoptsnum - 1));

	weightsMat = Mat::zeros(clustersMat.size(), CV_32F);
	labelMat = Mat::zeros(imgMat.size(), CV_8U);
	Mat diffMat, sortidxMat, countMat, tempMat;
	Point minLoc;
	for (ii = 0; ii < imgcols; ii++)
	{
		diffMat = abs(repeat(imgMat.col(ii), 1, clusternum) - repeat(clustersMat.col(ii).t(), imgrows, 1));
		sortIdx(diffMat, sortidxMat, SORT_EVERY_ROW + SORT_ASCENDING);

		for (jj = 0; jj < imgrows; jj++)
		{
			minMaxLoc(sortidxMat.row(jj), 0, 0, &minLoc, 0);
			if (minLoc.x < clusternum/5)
				labelMat.at<uchar>(jj, ii) = 1;
		}

		tempMat = (sortidxMat < 1) / 255;
		tempMat.convertTo(tempMat, CV_32F);
		reduce(tempMat, countMat, 0, CV_REDUCE_SUM);
		
		countMat = countMat / imgrows;
		tempMat = countMat.t();
		tempMat.copyTo(weightsMat.col(ii));
	}
	tempMat = (weightsMat > 0.1) / 255;
	tempMat.convertTo(tempMat, CV_32F);
	weightsMat = weightsMat.mul(tempMat);
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


//轮廓按照面积大小降序排序
bool ascendSort(vector<Point> a, vector<Point> b) 
{ 
	return a.size() < b.size(); 
} 

//轮廓按照面积大小降序排序
bool descendSort(vector<Point> a, vector<Point> b) 
{	
	return a.size() > b.size();
}


void getDefectSize(Mat& Img, double& width, double& height)
{
	vector< vector< Point> > contours;  //用于保存所有轮廓信息	
	findContours(Img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);	

	//轮廓按照面积从小到大进行排序
	cv::drawContours(Img, contours, -1, Scalar(255, 0, 0), CV_FILLED);
	
	if (contours.size() != 0)
		{
			sort(contours.begin(), contours.end(), descendSort);//升序排序	
			vector<vector<Point> >::iterator itc = contours.begin();
			Rect rect = boundingRect(*itc);
			width = rect.width;
			height = rect.height;
		}	
}

int main(int argc, char* argv[])
{ 
	std::vector< std::string > xlm_list;
	std::vector< std::string > ownname_list;
	
	

	double t0 = (double)getTickCount();
	int k = 1;
	Mat img;
	
	for (k= 0; k < 1; k++)
	{
		img = imread("/home/xuesong/RailInspection/defectsImages/diaokuai/d5.png");
		resize(img, img, Size(200, 1000), 0, 0, CV_INTER_LINEAR);
		//img = imread("/home/xuesong/RailInspection/defectsImages/yulinshang/y3.png");
		//img = imread("/home/xuesong/RailInspection/defectsImages/test/1.jpg");
		//img = img(Rect(250, 0, 180, 480));
		
		if (img.empty())
			return -1;
		if (img.channels() != 1)
		{
			cvtColor(img, img, CV_BGR2GRAY);
		}
		if (img.depth() == CV_8U)
		{
			img.convertTo(img, CV_32F);
		}

		// image normalization
		//resize(img, img, Size(200, 1000), 0, 0, CV_INTER_LINEAR);
		log_normalization(img);
		//imshow("gray", img);
		//waitKey();

		int m_rows = img.rows;
		int m_cols = img.cols;

		Mat LocationPrior(1, m_cols, CV_32F);
		float* lpData = (float*)LocationPrior.data;
		float tempV;
		for (int i = 0; i < m_cols; i++)
		{
			tempV = (float)i / (float)(m_cols - 1);
			*lpData = 0.4 * tempV * (1 - tempV);
			lpData++;
		}
		Mat LocationPriorMat;
		repeat(LocationPrior, m_rows, 1, LocationPriorMat);

		Mat clustersMat, weightsMat, labelMat;
		mean_shift(img, clustersMat, weightsMat, labelMat);

		Mat tempMat;
		int ii, jj;
		Mat diffMat[3];
		Mat saliencyMat = Mat::zeros(img.size(), CV_32F);
		for (ii = 0; ii < m_cols; ii++)
		{
			jj = max(ii - 1, 0);
			diffMat[0] = abs(repeat(img.col(ii), 1, clusternum) - repeat(clustersMat.col(jj).t(), m_rows, 1));
			diffMat[0] = diffMat[0].mul(repeat(weightsMat.col(jj).t(), m_rows, 1));

			jj = ii;
			diffMat[1] = abs(repeat(img.col(ii), 1, clusternum) - repeat(clustersMat.col(jj).t(), m_rows, 1));
			diffMat[1] = diffMat[0].mul(repeat(weightsMat.col(jj).t(), m_rows, 1));

			jj = min(ii + 1, m_cols - 1);
			diffMat[2] = abs(repeat(img.col(ii), 1, clusternum) - repeat(clustersMat.col(jj).t(), m_rows, 1));
			diffMat[2] = diffMat[0].mul(repeat(weightsMat.col(jj).t(), m_rows, 1));

			tempMat = (diffMat[0] + diffMat[1] + diffMat[2]) / 3.0;
			reduce(tempMat, saliencyMat.col(ii), 1, CV_REDUCE_SUM);
		}
		saliencyMat = saliencyMat.mul(LocationPriorMat);

		tempMat = (saliencyMat > thr_segment) / 255;
		labelMat = labelMat.mul(tempMat);
		
		cv::Mat opened;
		Mat image = labelMat*255;
		//cv::imshow("origin", image);

		//do open process for the image
		cv::Mat element5(4, 4, CV_8U, cv::Scalar(1));
		cv::morphologyEx(image, opened, cv::MORPH_OPEN, element5);

		double width = 0.;
		double height = 0.;
		//getDefectSize(opened, width, height);
		//cout << "the area of defect: " << width * height << endl;
		//cv::imshow("result",opened);
		
	
		//cv::morphologyEx(image, closed, cv::MORPH_CLOSE, element5);

		//cv::imshow("closed",closed);
		//waitKey();

		// save result
		//imwrite("results2.jpg", labelMat*255);
		//imwrite("results/origin2.jpg", image);
		//imwrite("results/opened2.jpg", opened);
		//imwrite("results/closed2.jpg", closed);
	}

	t0 = (double)getTickCount() - t0;
	cout << " time:\n " << t0 * 1000 / getTickFrequency() / k << " ms" << endl;
	waitKey();
	return 0;
}

