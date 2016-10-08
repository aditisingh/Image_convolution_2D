#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>

using namespace std;

struct  pixel //to store RGB values
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

pixel padding(pixel** Pixel_val, int x_coord, int y_coord, int img_width, int img_height) //padding the image,depending on pixel coordinates
//can be replaced by reflect for better result //currently zero padding
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
	time_t start_of_code=time(NULL);
	if(argc != 3) //there should be three arguments
	return 1; //exit and return an error
	
	float sigma = atof(argv[2]); //standard deviation for the gaussian 
    
	//Getting the kernel
	int k=floor(6*sigma);//sigma might have fractional part

	if(k%2==0) k++; //to make the size odd

	float **kernel0 = (float **)malloc(k * sizeof(float*)); //y based gaussian
	float **kernel1 = (float **)malloc(1* sizeof(float*));	//x based gaussian

	for(int i=0;i<k;i++)
		kernel0[i]=(float*)malloc(1*sizeof(float));
	
	kernel1[0]=(float*)malloc(k*sizeof(float));

	float constant1=sqrt(2*M_PI*sigma*sigma);//constants needed to define the kernel
	float constant2=2*sigma*sigma;

	int mid=floor(k/2);
	kernel0[mid][0]=1/constant1;
	kernel1[0][mid]=1/constant1;

	for(int i=0;i<floor(k/2);i++)	//using symmetry from center, to generate the separable kernels 
	{
		kernel0[i][0]=((exp(-(floor(k/2)-i)*(floor(k/2)-i)/constant2)))/constant1;

		kernel1[0][i]=kernel0[i][0];

		kernel0[k-1-i][0]=kernel0[i][0];

		kernel1[0][k-1-i]=kernel1[0][i];

	}
	time_t kernel_generation=time(NULL); //find time taken for kernel generation

	
	//reading the PPM file line by line
	ifstream infile;
	infile.open(argv[1]);
	string line;

	int img_wd, img_ht;
	int max_val;
	int line_count=0;

	//line one contains P6, line 2 mentions about gimp version, line 3 stores the height and width
	getline(infile, line);
	istringstream iss1(line);

	//reading first line to check format
	int word;
	string str1;

	iss1>>str1;
	if(str1.compare("P6")!=0)	//comparing magic number
	{
		cout<<"wrong file format"<<endl;
		return 1;
	}

	getline(infile,line); //this line has version related comment, hence ignoring

	getline(infile,line); //this stores image dims
	istringstream iss2(line);
	iss2>>word;// this will be image width
	img_wd=word;
	iss2>>word;// this will be image height
	img_ht=word;

	//storing the pixels as 2d images
	pixel **Pixel = (pixel**)malloc((img_ht)*sizeof(pixel*));
	pixel **Pixel_tmp = (pixel **)malloc((img_ht) * sizeof(pixel*)); 
	
	for(int i=0;i<(img_ht);i++){
		Pixel_tmp[i]=(pixel*)malloc(img_wd*sizeof(pixel));
		Pixel[i]=(pixel*)malloc((img_wd)*sizeof(pixel));}



	int pix_cnt=0, cnt=0, row,col;

	getline(infile,line); //this stores max value
	
	istringstream iss3(line);
	iss3>>word;
	max_val=word;//max pixel value

	unsigned int val;

	while (getline(infile, line))
	{
		istringstream iss4(line);
		for (int i=0; i<=line.length();i++)
		{
			if(pix_cnt<img_ht*img_wd)
			{	
				val =((int)line[i]);
				row=floor(pix_cnt/img_wd);
				col=pix_cnt%img_wd;
				
				if(cnt%3==0)
				{		
					Pixel[row][col].r=val;
				}
				else if(cnt%3==1)
				{
					Pixel[row][col].g=val;
				}
				else
				{
					Pixel[row][col].b=val;
					pix_cnt++;
				}
				cnt++;
			}
		} 	
		line_count++;		
	}

	time_t reading_file=time(NULL);


	//perform convolution

	
	float tmp_r, tmp_g, tmp_b;

	//vertical convolution
	for(int i=0; i<img_wd;i++ )
	{		
		for(int j=0;j<img_ht;j++)
		{
			tmp_r=0, tmp_g=0, tmp_b=0;
			for(int l=0;l<k;l++)
			{	
				pixel pix_val=padding(Pixel, i, j+l-(k-1)/2, img_wd, img_ht);
				tmp_r+=pix_val.r * kernel0[l][0];
				tmp_b+=pix_val.b * kernel0[l][0];
				tmp_g+=pix_val.g * kernel0[l][0];
			}
			Pixel_tmp[j][i].r=tmp_r;
			Pixel_tmp[j][i].g=tmp_g;
			Pixel_tmp[j][i].b=tmp_b;
		}
	}
	
	time_t vertical_convolution=time(NULL);

	//horizontal convolution
	for(int j=0;j<img_ht;j++ )
	{
		for(int i=0; i<img_wd;i++)
		{
			tmp_r=0, tmp_g=0, tmp_b=0;
			for(int l=0; l<k;l++)
			{
				pixel pix_val=padding(Pixel_tmp, i+l-(k-1)/2, j, img_wd, img_ht);
				tmp_r+=pix_val.r * kernel1[0][l];
				tmp_g+=pix_val.g * kernel1[0][l];
				tmp_b+=pix_val.b * kernel1[0][l];
			}
			Pixel[j][i].r=tmp_r;
			Pixel[j][i].g=tmp_g;
			Pixel[j][i].b=tmp_b;
		
		}
	}
	time_t horizontal_convolution=time(NULL);


	//writing this to PPM file
	ofstream ofs;
	ofs.open("output.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_val<<"\n";
	
	for(int j=0; j <img_ht;j++)
	{
		for (int i=0; i<img_wd;i++)
		{
			ofs<<Pixel[j][i].r<<Pixel[j][i].g<<Pixel[j][i].b;	//write as ascii
		}
	}
	
	ofs.close();
	time_t end=time(NULL);

	//display time taken for different processes
	cout<<" Total execution time: "<<double(end-start_of_code)<<" sec"<<endl;
	cout<<" Saving the result:"<<double(end-horizontal_convolution)<<" sec"<<endl;
	cout<<" horizontal convolution time:" <<double(horizontal_convolution-vertical_convolution)<<" sec"<<endl;
	cout<<" vertical_convolution time: "<<double(vertical_convolution - reading_file)<<"sec"<<endl;
	cout<<" File reading time:"<<double(reading_file - kernel_generation)<<" sec"<<endl;
	cout<<" Kernel generation time:"<<double(kernel_generation - start_of_code)<<" sec"<<endl;
	return 0;
}


