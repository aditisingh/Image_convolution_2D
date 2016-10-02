#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>

using namespace std;

struct  pixel
{
	unsigned int r;
	unsigned int g;
	unsigned int b;
};

int main(int argc, char* argv[])
{
	if(argc != 3) //there should be three arguments
	return 1; //exit and return an error
	
	int sigma = atoi(argv[2]); 
    
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

	getline(infile,line); //this stores max value

	istringstream iss3(line);
	iss3>>word;
	max_val=atoi(word.c_str());//max pixel value
	
	while (getline(infile, line))
	{
		istringstream iss4(line);
		while(iss4>>word)
		{
			for (int i=0; i<word.size();i++)
			{
				int val =(unsigned int)word[i];
				// cout<<val;
			}
		}
		line_count++;
	}
	cout<<endl<<line_count<<endl;
	cout<<max_val<<endl; cout<<img_ht<<endl; cout<<img_wd<<endl;
	// cout<<"ends";
	return 0;
}