#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

struct  pixel
{
	unsigned int r;
	unsigned int g;
	unsigned int b;
};

pixel padding(pixel** Pixel_val, int x_coord, int y_coord, int img_width, int img_height) //that's for giving pixel values channel wise
{
	pixel Px;
	Px.r=0; Px.g=0; Px.b=0;
	if(x_coord>=img_width)
		return Px;
	else if(x_coord<0)
		return Px;
	else if(y_coord>=img_height)
		return Px;
	else if(y_coord<0)
		return Px;
	else	
		return Pixel_val[y_coord][x_coord];
}

int main(int argc, char* argv[])
{
	if(argc != 3) //there should be three arguments
	return 1; //exit and return an error
	
	int sigma = atoi(argv[2]); 
    
	//Getting the kernel
	int k=floor(6*sigma);//sigma might have fractional part

	if(k%2==0) k++; //odd k

	float **kernel0 = (float **)malloc(k * sizeof(float*)); //y based gaussian
	float **kernel1 = (float **)malloc(1* sizeof(float*));	//x based gaussian

	for(int i=0;i<k;i++)
		kernel0[i]=(float*)malloc(1*sizeof(float));
	
	kernel1[0]=(float*)malloc(k*sizeof(float));

	float constant=sqrt(2*M_PI*sigma*sigma);

	int mid=floor(k/2);
	kernel0[mid][0]=1/constant;
	kernel1[0][mid]=1/constant;

	for(int i=0;i<floor(k/2);i++)	//using symmetry from center
	{
		kernel0[i][0]=(exp(-(floor(k/2)-i)*(floor(k/2)-i)))/constant;

		kernel1[0][i]=kernel0[i][0];

		kernel0[k-1-i][0]=kernel0[i][0];

		kernel1[0][k-1-i]=kernel1[0][i];

	}

	/*for(int i=0; i<k;i++)
	{	
		for(int j=0;j<k;j++)
		cout<<kernel0[i][0]*kernel1[0][j]<<" ";
		cout<<endl;
	}*/ //Verified kernel implementation

	//int p=(k-1)/2;		//padding


	//reading the PPM file line by line
	ifstream infile;
	infile.open(argv[1]);
	string line;

	int img_wd, img_ht;
	int max_val;
	int line_count=0;
	int numpixels;

	//line one contains P6, line 2 mentions about gimp version, line 3 stores the height and width
	getline(infile, line);
	istringstream iss1(line);

	//reading first line to check format
	string word;
	iss1>>word;
	if(word.compare("P6")!=0)	
	{
		cout<<"wrong file format"<<endl;
		return 1;
	}

	getline(infile,line); //this line has version related comment, hence ignoring

	getline(infile,line); //this stores image dims
	istringstream iss2(line);
	iss2>>word;// this will be image width
	img_wd=atoi(word.c_str());
	iss2>>word;// this will be image height
	img_ht=atoi(word.c_str());
	numpixels=img_wd*img_ht;


	//storing the pixels as 2d images
	/*pixel **Pixel = (pixel**)malloc((img_ht+2*p)*sizeof(pixel*));
	
	for(int i=0;i<(img_ht+2*p);i++)
	Pixel[i]=(pixel*)malloc((img_wd+2*p)*sizeof(pixel));

	for(int i=0;i<img_wd+2*p;i++)
	{
		for(int j=0;j<img_ht+2*p;j++)
		{
			Pixel[j][i].r=0;
			Pixel[j][i].g=0;
			Pixel[j][i].b=0;
		}
	}
	*/
	pixel **Pixel = (pixel**)malloc((img_ht)*sizeof(pixel*));
	
	for(int i=0;i<(img_ht);i++)
	Pixel[i]=(pixel*)malloc((img_wd)*sizeof(pixel));

	


	unsigned int pix_cnt=0;	
	getline(infile,line); //this stores max value
	unsigned int cnt=0;
	unsigned int r,c;
	istringstream iss3(line);
	iss3>>word;
	max_val=atoi(word.c_str());//max pixel value

	while (getline(infile, line))
	{
		istringstream iss4(line);
		
		for (int i=0; i<=line.length();i++)
		{
			if(pix_cnt<img_ht*img_wd)
			{	
				unsigned int val =(unsigned int)line[i];
				r=floor(pix_cnt/img_wd);
				c=pix_cnt%img_wd;
				//cout<<r<<" "<<c<<" "<<img_ht<<" "<<img_wd<<" "<<pix_cnt<<endl;
				if(cnt%3==0)
				{		
					Pixel[r][c].r=val;//Pixel[r+p][c+p].r=val;
					// cout<<val<<" ";
				}
				if(cnt%3==1)
				{
					Pixel[r][c].g=val;
				}
				if(cnt%3==2)
				{
					Pixel[r][c].b=val;
					pix_cnt++;
				}
				cnt++;
			}	
		} 
		line_count++;		
	}
	
	//cout<<Pixel[img_ht-1][img_wd-1].r<<" "<<Pixel[img_ht-1][img_wd-1].g<<" "<<Pixel[img_ht-1][img_wd-1].b<<endl;
	//Pixels have been stored successfully



	//perform convolution

	//pixel **Pixel_res1 = (pixel **)malloc((img_ht+2*p) * sizeof(pixel*)); 
	pixel **Pixel_tmp = (pixel **)malloc((img_ht) * sizeof(pixel*)); 
	
	for(int i=0;i<(img_ht);i++)
		Pixel_tmp[i]=(pixel*)malloc(img_wd*sizeof(pixel));

	//vertical convolution
	/*for(int j=0;j<(img_ht);j++)
	{		
		for(int i=0; i<img_wd;i++)
		{
			float tmp_r=0, tmp_g=0, tmp_b=0;
			for(int l=-(k-1)/2;l<=(k-1)/2;l++)
			{
				pixel pix_val=padding(Pixel, i, j+l, img_wd, img_ht);
				tmp_r+=pix_val.r * kernel0[l+(k-1)/2][0];
				tmp_b+=pix_val.b * kernel0[l+(k-1)/2][0];
				tmp_g+=pix_val.g * kernel0[l+(k-1)/2][0];
			}
			Pixel_tmp[j][i].r=tmp_r;
			Pixel_tmp[j][i].g=tmp_g;
			Pixel_tmp[j][i].b=tmp_b;
		
		}
	}

	//horizontal convolution
	for(int i=0; i<img_wd;i++)
	{
		for(int j=0;j<img_ht;j++)
		{
			float tmp_r=0, tmp_g=0, tmp_b=0;
			for(int l=-(k-1)/2; l<=(k-1)/2;l++)
			{
				pixel pix_val=padding(Pixel_tmp, i+l, j, img_wd, img_ht);
				tmp_r+=pix_val.r * kernel1[0][l+(k-1)/2];
				tmp_b+=pix_val.b * kernel1[0][l+(k-1)/2];
				tmp_g+=pix_val.g * kernel1[0][l+(k-1)/2];
			}
			Pixel[j][i].r=tmp_r;
			Pixel[j][i].g=tmp_g;
			Pixel[j][i].b=tmp_b;
		
		}
	}*/

	//writing this to PPM file
	ofstream ofs;
	ofs.open("output.ppm", ofstream::out);
	ofs<<"P6"<<endl;
	ofs<<"# File after convolution"<<endl;
	ofs<<img_wd<<" "<<img_ht<<endl;	//check if ASCII conversion is needed
	ofs<<max_val<<endl;
	
	for(int j=0; j <img_ht;j++)
	{
		for (int i=0; i<img_wd;i++)
		{
			ofs<<static_cast<char>(Pixel[j][i].r)<<static_cast<char>(Pixel[j][i].g)<<static_cast<char>(Pixel[j][i].b);	//write as ascii
		}
		ofs<<endl;
	}
	
	
	

	return 0;
}

