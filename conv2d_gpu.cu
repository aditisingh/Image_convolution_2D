#include <fstream>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime_api.h>

using namespace std;

struct  pixel //to store RGB values
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

static void HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		cout<<cudaGetErrorString(err)<<" in "<< file <<" at line "<< line<<endl;
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ void padding(pixel** Pixel_val, int x_coord, int y_coord, int img_width, int img_height, pixel Px) //padding the image,depending on pixel coordinates, can be replaced by reflect for better result //currently zero padding
{
	Px.r=0; Px.g=0; Px.b=0;
	if(x_coord<img_width && y_coord<img_height && x_coord>=0 && y_coord>=0)	
		Px=Pixel_val[y_coord][x_coord];
}


__global__ void vertical_conv(pixel** Pixel_in, pixel** Pixel_out,int img_wd, int img_ht, float** kernel, int k)
{
	float tmp_r, tmp_g, tmp_b;
	pixel pix_val;
	int row=blockIdx.y*blockDim.y + threadIdx.y;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if(row<img_ht && col<img_wd){
		tmp_r=0, tmp_g=0, tmp_b=0;
		for(int l=0;l<k;l++)
		{	
			
			padding(Pixel_in, col, row+l-(k-1)/2, img_wd, img_ht, pix_val);
			tmp_r+=pix_val.r * kernel[l][0];
			tmp_b+=pix_val.b * kernel[l][0];
			tmp_g+=pix_val.g * kernel[l][0];
		}

		Pixel_out[row][col].r=tmp_r;
		Pixel_out[row][col].g=tmp_g;
		Pixel_out[row][col].b=tmp_b;
	}
}

__global__ void horizontal_conv(pixel** Pixel_in, pixel** Pixel_out, int img_wd, int img_ht, float** kernel, int k)
{
	float tmp_r, tmp_b, tmp_g;
	pixel pix_val;

	//horizontal convolution
	int row=blockIdx.y*blockDim.y + threadIdx.y;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	tmp_r=0, tmp_g=0, tmp_b=0;
	if(row<img_ht && col<img_wd)
	{
		for(int l=0; l<k;l++)
		{
			padding(Pixel_in, col+l-(k-1)/2, row, img_wd, img_ht, pix_val);
			tmp_r+=pix_val.r * kernel[0][l];
			tmp_g+=pix_val.g * kernel[0][l];
			tmp_b+=pix_val.b * kernel[0][l];
		}
		Pixel_out[row][col].r=tmp_r;
		Pixel_out[row][col].g=tmp_g;
		Pixel_out[row][col].b=tmp_b;
	}
}

int main(int argc, char* argv[])
{
	int nDevices;
	HANDLE_ERROR(cudaGetDeviceCount(&nDevices));
	cout<<"number of devices="<<nDevices<<endl;

	for(int i=0;i<nDevices;i++){
	cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    /*printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    cout<<"  Total global memory :"<<prop.totalGlobalMem<<endl;
    cout<<"  Shared memory per block :"<<prop.sharedMemPerBlock<<endl;
    cout<<"  Regs per block :"<<prop.regsPerBlock<<endl;
    cout<<"  Warp size :"<<prop.warpSize<<endl;
    cout<<"  Max threads per block :"<<prop.maxThreadsPerBlock<<endl;
    cout<<"  Max threads dimension :"<<prop.maxThreadsDim[0]<<" "<<prop.maxThreadsDim[1]<<" "<<prop.maxThreadsDim[2]<<endl;
    cout<<"  Max grid size: "<<prop.maxGridSize[0]<<" "<<prop.maxThreadsDim[1]<<" "<<prop.maxThreadsDim[2]<<endl;
    */  
  }
	
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
	cout<<" Kernel generation time:"<<double(kernel_generation - start_of_code)<<" sec"<<endl;

	
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
	
	for(int i=0;i<(img_ht);i++)
	{
		Pixel_tmp[i]=(pixel*)malloc(img_wd*sizeof(pixel));
		Pixel[i]=(pixel*)malloc((img_wd)*sizeof(pixel));
	}



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
	cout<<" File reading time:"<<double(reading_file - kernel_generation)<<" sec"<<endl;


	cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));


    int thread_block=sqrt(prop.maxThreadsPerBlock);
	dim3 DimGrid(ceil(img_wd/thread_block),ceil(img_ht/thread_block),1);
	dim3 DimBlock(sqrt(prop.maxThreadsPerBlock),sqrt(prop.maxThreadsPerBlock),1);

	int *k_gpu, *img_wd_gpu, *img_ht_gpu;
	//allocating gpu memory

	pixel **Pixel_tmp_gpu, **Pixel_gpu;
    

	HANDLE_ERROR(cudaMalloc(&Pixel_tmp_gpu,img_wd*img_ht*sizeof(pixel)));
	HANDLE_ERROR(cudaMalloc(&Pixel_gpu,img_wd*img_ht*sizeof(pixel)));


	float **kernel0_gpu, **kernel1_gpu;

	HANDLE_ERROR(cudaMalloc(&kernel0_gpu,k *1*sizeof(float*)));
	HANDLE_ERROR(cudaMalloc(&kernel1_gpu,1*k*sizeof(float**)));

	
	HANDLE_ERROR(cudaMalloc(&k_gpu,sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&img_wd_gpu,sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&img_ht_gpu,sizeof(int)));

	cout<<"memory allocated"<<endl;

	//copying needed data

	HANDLE_ERROR(cudaMemcpy(Pixel_tmp_gpu,Pixel_tmp,img_wd*img_ht*sizeof(pixel),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Pixel_gpu,Pixel,img_wd*img_ht*sizeof(pixel),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(kernel0_gpu,kernel0,k*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(kernel1_gpu,kernel1,k*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(k_gpu,&k,sizeof(int),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(img_wd_gpu,&img_wd,sizeof(int),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(img_ht_gpu,&img_ht,sizeof(int),cudaMemcpyHostToDevice));

	cout<<"memory transfers done"<<endl;

	vertical_conv<<<DimGrid,DimBlock>>>(Pixel, Pixel_tmp,img_wd, img_ht,kernel0,k);
	time_t vertical_convolution=time(NULL);

	cout<<" vertical_convolution time: "<<double(vertical_convolution - reading_file)<<"sec"<<endl;

	HANDLE_ERROR(cudaMemcpy(Pixel_tmp,Pixel_tmp_gpu,img_wd*img_ht*sizeof(pixel),cudaMemcpyDeviceToHost));

	
	horizontal_conv<<<DimGrid,DimBlock>>>(Pixel_tmp, Pixel, img_wd, img_ht, kernel1, k);
	time_t horizontal_convolution=time(NULL);
	cout<<" horizontal convolution time:" <<double(horizontal_convolution-vertical_convolution)<<" sec"<<endl;
	HANDLE_ERROR(cudaMemcpy(Pixel,Pixel_gpu,img_wd*img_ht*sizeof(pixel),cudaMemcpyDeviceToHost));
/*

	//writing this to PPM file
	ofstream ofs;
	ofs.open("output_gpu.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_val<<"\n";
	
	for(int j=0; j <img_ht;j++)
	{
		for (int i=0; i<img_wd;i++)
		{
			ofs<<Pixel[j][i].r<<Pixel[j][i].g<<Pixel[j][i].b;	//write as ascii
		}
	}
	
	
	ofs.close();*/
	time_t end=time(NULL);
	cout<<" Saving the result:"<<double(end-horizontal_convolution)<<" sec"<<endl;
	
	HANDLE_ERROR(cudaFree(Pixel_gpu)); 
	HANDLE_ERROR(cudaFree(Pixel_tmp_gpu));
	HANDLE_ERROR(cudaFree(k_gpu));
	HANDLE_ERROR(cudaFree(kernel0_gpu));
	HANDLE_ERROR(cudaFree(kernel1_gpu));
	HANDLE_ERROR(cudaFree(img_ht_gpu));
	HANDLE_ERROR(cudaFree(img_wd_gpu));


	//display time taken for different processes
	cout<<" Total execution time: "<<double(end-start_of_code)<<" sec"<<endl;

	return 0;
}