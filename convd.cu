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

__host__ __device__ pixel padding(pixel* Pixel_val, int x_coord, int y_coord, int img_width, int img_height) 
{	pixel Px;
	Px.r=0; Px.g=0; Px.b=0;
	if(x_coord< img_width && y_coord <img_height && x_coord>=0 && y_coord>=0)
	{
		Px=Pixel_val[y_coord*img_width+x_coord];
	}
	return Px;
}

__global__ void vertical_conv(pixel* Pixel_in_v, pixel* Pixel_out_v,int img_wd_v, int img_ht_v, float* kernel_v, int k_v)
{
	float tmp_r, tmp_g, tmp_b;
	//int pix_idx_v=blockIdx.x*blockDim.x + threadIdx.x;
	//int row=(int)(pix_idx_v/img_wd_v);
	//int col=pix_idx_v%img_wd_v;
	size_t col=blockIdx.x*blockDim.x + threadIdx.x;
	size_t row=blockIdx.y*blockDim.y + threadIdx.y;
	size_t pix_idx_v=row*img_wd_v+col;
	tmp_r=0, tmp_g=0, tmp_b=0;		
	if(row<img_ht_v && col<img_wd_v){

		for(int l=0;l<k_v;l++)
		{//doing by 1 D arrays	
			pixel pix_val=padding(Pixel_in_v, col, (row+l-(k_v-1)/2), img_wd_v, img_ht_v);
			tmp_r+=pix_val.r * kernel_v[l];
			tmp_b+=pix_val.b * kernel_v[l];
			tmp_g+=pix_val.g * kernel_v[l];
		}

		Pixel_out_v[pix_idx_v].r=tmp_r;
		Pixel_out_v[pix_idx_v].g=tmp_g;
		Pixel_out_v[pix_idx_v].b=tmp_b;
	}
}			


__global__ void horizontal_conv(pixel* Pixel_in, pixel* Pixel_out, int img_wd, int img_ht, float* kernel, int k)
{
	float tmp_r, tmp_b, tmp_g;
	//horizontal convolution
	//int pix_idx=blockIdx.x*blockDim.x + threadIdx.x;
	//int row=(int)(pix_idx/img_wd);
	//int col=pix_idx%img_wd;
	size_t col=blockIdx.x*blockDim.x + threadIdx.x;
	size_t row=blockIdx.y*blockDim.y + threadIdx.y;
	size_t pix_idx=row*img_wd+col;

	tmp_r=0, tmp_g=0, tmp_b=0;
	if(row<img_ht && col<img_wd)
	{
		for(int l=0; l<k;l++)
		{
			pixel pix_val=padding(Pixel_in, col+ l-(k-1)/2, row, img_wd, img_ht);
			tmp_r+=pix_val.r * kernel[l];
			tmp_g+=pix_val.g * kernel[l];
			tmp_b+=pix_val.b * kernel[l];
		}
		Pixel_out[pix_idx].r=tmp_r;
		Pixel_out[pix_idx].g=tmp_g;
		Pixel_out[pix_idx].b=tmp_b;
	}
}
pixel* vertical_conv_cpu(pixel* Pixel_in, int img_wd, int img_ht, float* kernel, int k)
{
	float tmp_r, tmp_g, tmp_b;
	pixel* Pixel_out=(pixel*)(malloc(img_wd*img_ht*sizeof(pixel)));
	for(int pix_idx=0;pix_idx<img_ht*img_wd;pix_idx++)
	{
		//vertical convolution
		int row=(int)(pix_idx/img_wd);
		int col=pix_idx%img_wd;
					
		if(row<img_ht && col<img_wd){
			tmp_r=0, tmp_g=0, tmp_b=0;
			for(int l=0;l<k;l++)
			{//doing by 1 D arrays	
				pixel pix_val=padding(Pixel_in, col,(row+l-(k-1)/2), img_wd, img_ht);
				tmp_r+=pix_val.r * kernel[l];
				tmp_b+=pix_val.b * kernel[l];
				tmp_g+=pix_val.g * kernel[l];
			}

			Pixel_out[pix_idx].r=tmp_r;
			Pixel_out[pix_idx].g=tmp_g;
			Pixel_out[pix_idx].b=tmp_b;
		}
	}
	return Pixel_out;
}

pixel* horizontal_conv_cpu(pixel* Pixel_in, int img_wd, int img_ht, float* kernel, int k )
{
	float tmp_r, tmp_b, tmp_g;
	//horizontal convolution
	pixel* Pixel_out=(pixel*)(malloc(img_wd*img_ht*sizeof(pixel)));

	for(int pix_idx=0;pix_idx<img_ht*img_wd;pix_idx++)
	{
		//vertical convolution
		int row=(int)(pix_idx/img_wd);
		int col=pix_idx%img_wd;
					
		if(row<img_ht && col<img_wd){
			tmp_r=0, tmp_g=0, tmp_b=0;
			for(int l=0;l<k;l++)
			{//doing by 1 D arrays	
				pixel pix_val=padding(Pixel_in, col+l-(k-1)/2,row, img_wd, img_ht);
				tmp_r+=pix_val.r * kernel[l];
				tmp_b+=pix_val.b * kernel[l];
				tmp_g+=pix_val.g * kernel[l];
			}

			Pixel_out[pix_idx].r=tmp_r;
			Pixel_out[pix_idx].g=tmp_g;
			Pixel_out[pix_idx].b=tmp_b;
		}
	}
	return Pixel_out;
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

	float *kernel0 = (float *)malloc(k * sizeof(float)); //y based gaussian
	float *kernel1 = (float *)malloc(k * sizeof(float));	//x based gaussian

	
	float constant1=sqrt(2*M_PI*sigma*sigma);//constants needed to define the kernel
	float constant2=2*sigma*sigma;

	int mid=floor(k/2);
	kernel0[mid]=1/constant1;
	kernel1[mid]=1/constant1;

	for(int i=0;i<floor(k/2);i++)	//using symmetry from center, to generate the separable kernels 
	{
		kernel0[i]=((exp(-(floor(k/2)-i)*(floor(k/2)-i)/constant2)))/constant1;

		kernel1[i]=kernel0[i];

		kernel0[k-1-i]=kernel0[i];

		kernel1[k-1-i]=kernel1[i];

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
	//cout<<"wd="<<img_wd<<", ht="<<img_ht<<endl;

	size_t num_pixels=img_wd*img_ht;
	pixel *Pixel_out=(pixel*)malloc(num_pixels*sizeof(pixel));
	//storing the pixels as lexicographically
	pixel *Pixel = (pixel*)malloc(num_pixels*sizeof(pixel));

	int pix_cnt=0, cnt=0;

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
			if(pix_cnt<num_pixels)
			{	
				val =((int)line[i]);
				
				if(cnt%3==0)
				{		
					Pixel[pix_cnt].r=val;
				}
				else if(cnt%3==1)
				{
					Pixel[pix_cnt].g=val;
				}
				else
				{
					Pixel[pix_cnt].b=val;
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

    	float thread_block=sqrt(prop.maxThreadsPerBlock);
	dim3 DimGrid(ceil(img_wd/thread_block),ceil(img_ht/thread_block),1);
	dim3 DimBlock(thread_block,thread_block,1);
	cout<<"grid="<<DimGrid.x<<" "<<DimGrid.y<<" "<<DimGrid.z<<endl;
	cout<<"block="<<DimBlock.x<<" "<<DimBlock.y<<" "<<DimBlock.z<<endl;
	//allocating gpu memory

	pixel *Pixel_tmp_gpu, *Pixel_gpu, *Pixel_gpu_res;
   
	HANDLE_ERROR(cudaMalloc(&Pixel_gpu_res,num_pixels*sizeof(pixel))); //allocate space to store convolution result
	HANDLE_ERROR(cudaMemset(Pixel_gpu_res,128,num_pixels*sizeof(pixel)));
	HANDLE_ERROR(cudaMalloc(&Pixel_tmp_gpu,num_pixels*sizeof(pixel))); //allocate space to store convolution temporary

	HANDLE_ERROR(cudaMalloc(&Pixel_gpu,num_pixels*sizeof(pixel))); //allocate space to copy image to GPU memory
	
	float *kernel0_gpu, *kernel1_gpu;
	

	HANDLE_ERROR(cudaMalloc(&kernel0_gpu, k*sizeof(float)));//allocate memory for kernel0

	HANDLE_ERROR(cudaMalloc(&kernel1_gpu, k*sizeof(float)));//allocate memory for kernel1
	
	cout<<"memory allocated"<<endl;

	//copying needed data

	HANDLE_ERROR(cudaMemcpy(Pixel_gpu, Pixel, num_pixels*sizeof(pixel),cudaMemcpyHostToDevice));//copy input image from global to gpu

	HANDLE_ERROR(cudaMemcpy(kernel0_gpu, kernel0,k*sizeof(float),cudaMemcpyHostToDevice));//copy the kernel0 host to device

	HANDLE_ERROR(cudaMemcpy(kernel1_gpu,kernel1,k*sizeof(float),cudaMemcpyHostToDevice));//copy kernel1 host to device

	cout<<"memory transfers done"<<endl;

	vertical_conv<<<DimGrid,DimBlock>>>(Pixel_gpu, Pixel_tmp_gpu,img_wd, img_ht,kernel0_gpu,k);
	cout<<img_wd<<" "<<img_ht<<endl;
	time_t vertical_convolution=time(NULL);

	cout<<" vertical_convolution time: "<<double(vertical_convolution - reading_file)<<"sec"<<endl;

	horizontal_conv<<<DimGrid,DimBlock>>>(Pixel_tmp_gpu, Pixel_gpu_res, img_wd, img_ht, kernel1_gpu, k);
	time_t horizontal_convolution=time(NULL);

	HANDLE_ERROR(cudaMemcpy(Pixel_out,Pixel_gpu_res, num_pixels*sizeof(pixel),cudaMemcpyDeviceToHost));

	cout<<" horizontal convolution time:" <<double(horizontal_convolution-vertical_convolution)<<" sec"<<endl;
	//writing this to PPM file
	ofstream ofs;
	ofs.open("output_gpu.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_val<<"\n";
	
	for(int i=0; i <num_pixels;i++)
	{
		ofs<<Pixel_out[i].r<<Pixel_out[i].g<<Pixel_out[i].b;	//write as ascii
	}
	ofs.close();
	
	time_t end1=time(NULL);

	pixel *Pixel_tmp=vertical_conv_cpu(Pixel,img_wd, img_ht,kernel0,k);
	
	time_t vertical_convolution=time(NULL);

	Pixel_out=horizontal_conv_cpu(Pixel_tmp, img_wd, img_ht, kernel1, k);

	time_t horizontal_convolution=time(NULL);


	//writing this to PPM file

	ofs.open("output.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_val<<"\n";
	
	for(int j=0; j <img_ht*img_wd;j++)
	{
		ofs<<Pixel_out[j].r<<Pixel_out[j].g<<Pixel_out[j].b;	//write as ascii
	}
	
	ofs.close();
	time_t end=time(NULL);

		//cout<<" Saving the result:"<<double(end-horizontal_convolution)<<" sec"<<endl;

	//display time taken for different processes
	cout<<" Total execution time: "<<double(end-start_of_code)<<" sec"<<endl;

	return 0;
}
