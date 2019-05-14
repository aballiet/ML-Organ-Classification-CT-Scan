#include <set>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
using std::set ;
using std::cout ;
using std::cerr ;
using std::pair ;
using std::string ;
using std::vector ;
using std::ostream ;
using std::ifstream ;
using std::ofstream ;
using std::istringstream ;

#include <itkImage.h>
#include <itkMetaImageIO.h>
#include <itkNiftiImageIO.h>
#include <itkImageFileReader.h>
#include <itkStatisticsImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>

void printSyntax(ostream& os)
{
    os << "Usage:\n" ;
    os << "ExtractBlocks -i image -s segfiles [-b 33] [-f Imin] [-d 2] [-n 500] [-8] [-o prefix]\n" ;
    os << "ExtractBlocks -i image -s segfiles -m mapfiles [-t 0.75] [-r 0.5] [-b 33] [-f Imin] [-d 2] [-n 500] [-8] [-o prefix]\n" ;
	os << "  -i image\n" ;
	os << "    input NIFTI image\n" ;
	os << "  -s segfiles\n" ;
	os << "    filenames of associated grounth-truth segmentations\n" ;
	os << "  -b blocksize\n" ;
	os << "    size of extracted block; positive integer\n" ;
	os << "      33 by default\n" ;
	os << "  -f fill\n" ;
	os << "    fill value for blocks overlapping image boundary\n" ;
	os << "      image minimum intensity, by default\n" ;	
	os << "  -d dim\n" ;
	os << "    dimension of extracted block (rectangle: 2 or cube: 3)\n" ;
	os << "      2 by default\n" ;
	os << "  -n count\n" ;
	os << "    maximum number of extracted blocks per class; positive integer\n" ;
	os << "      500 by default\n" ;
	os << "  -8\n" ;
	os << "    force 8-bit output\n" ;
	os << "      input image range is left intact by default\n" ;    
    os << "  -m mapfiles\n" ;
	os << "    filenames of associated probability maps; if given, the program generates separate files for training and testing\n" ;    
	os << "  -t threshold\n" ;
	os << "    used to select voxels from probability maps; positive real (0,1], only meaningful if -m is given\n" ;
	os << "      0.75 by default\n" ;
	os << "  -r ratio\n" ;
	os << "    ratio of testing set to training set vectors; positive real (0,1], only meaningful if -m is given\n" ;
	os << "      0.5 by default\n" ;
	os << "  -o prefix\n" ;
	os << "    prefix of output filename\n" ;
	os << "      input image filename by default\n" ;  
}


int main(int argc, char* argv[])
{
	//////////////////// INPUT PARSING AND VALIDATION

    int blockDim = 2 ;
	int blockSize = 33 ;    
    int maxBlocks = 500 ;    
    float fillValue = 0 ; // will be set later
	double threshold = 0.75 ;    
    double testTrainRatio = 0.5 ;
    
    string inputFilename, outputPrefix ;
    vector<string> segmentationFilenames, mapFilenames ;
    
    bool force8bitOutput = false ;
    bool mapFilenamesSpecified = false ;    
	
	vector<int> radLexIndices ;

	// Parse and validate input
	vector<string> arguments ;
	for (int i = 1 ; i < argc ; ++i)
		arguments.push_back(argv[i]) ;
	
	vector<string>::iterator itOptI = find(arguments.begin(), arguments.end(), "-i") ;
	if (itOptI != arguments.end()) {
		vector<string>::iterator itArgI = itOptI ;
		++itArgI ;
		if ((itArgI == arguments.end()) || ((*itArgI)[0] == '-')) {
			cerr << "Input image argument missing; terminating.\n" ;
			return 2 ;
		}
		else {
			inputFilename = *itArgI ;
			arguments.erase(itArgI) ;
			itOptI = find(arguments.begin(), arguments.end(), "-i") ;
		}		
		arguments.erase(itOptI) ;
	}	
	else {
		cerr << "Required input image option missing; terminating.\n" ;
		printSyntax(cerr) ;
		return 1 ;		
	}
	
	vector<string>::iterator itOptS = find(arguments.begin(), arguments.end(), "-s") ;
	if (itOptS != arguments.end()) {
		vector<string>::iterator itArgS = itOptS ;
		++itArgS ;
		int argumentCounter = 0 ; 
		while ((itArgS != arguments.end()) && ((*itArgS)[0] != '-')) {
			segmentationFilenames.push_back(*itArgS) ;
			++itArgS ;
			++argumentCounter ;
		}
		if (!segmentationFilenames.size()) {
			cerr << "No segmentations were provided; terminating.\n" ;
			return 4 ;
		}
		else {
			itArgS = itOptS ;
			++itArgS ;
			arguments.erase(itArgS, itArgS + argumentCounter) ;
			itOptS = find(arguments.begin(), arguments.end(), "-s") ;
			arguments.erase(itOptS) ;
		}
	}	
	else {
		cerr << "Required segmentation images option missing; terminating.\n" ;
		return 3 ;		
	}
	
	vector<string>::iterator itOptB = find(arguments.begin(), arguments.end(), "-b") ;
	if (itOptB != arguments.end()) {
		vector<string>::iterator itArgB = itOptB ;
		++itArgB ;
		if ((itArgB == arguments.end()) || ((*itArgB)[0] == '-')) {
			cerr << "Block size argument missing; using default " << blockSize << ".\n" ;
		}
		else {
			int b ;
			istringstream is(*itArgB) ;
			is >> b ;
			if (!is)
				cerr << "Bad specification of block size argument: " << *itArgB << "; using default " << blockSize << ".\n" ;
			else {
				if (!((b > 0) && (b % 2)))
					cerr << "Invalid block size argument: " << b << "; using default " << blockSize << ".\n" ;
				else
					blockSize = b ;
			}
			arguments.erase(itArgB) ;
			itOptB = find(arguments.begin(), arguments.end(), "-b") ;
		}		
		arguments.erase(itOptB) ;
	}
	
	vector<string>::iterator itOptF = find(arguments.begin(), arguments.end(), "-f") ;
	if (itOptF != arguments.end()) {
		vector<string>::iterator itArgF = itOptF ;
		++itArgF ;
		if ((itArgF == arguments.end()) || ((*itArgF)[0] == '-')) {
			cerr << "Fill value argument missing; using image minimum intensity.\n" ;
		}
		else {
			float f ;
			istringstream is(*itArgF) ;
			is >> f ;
			if (!is)
				cerr << "Bad specification of fill value argument: " << *itArgF << "; using image minimum intensity.\n" ;
			else
				fillValue = f ;
			arguments.erase(itArgF) ;
			itOptF = find(arguments.begin(), arguments.end(), "-f") ;
		}		
		arguments.erase(itOptF) ;
	}	
	
	vector<string>::iterator itOptD = find(arguments.begin(), arguments.end(), "-d") ;
	if (itOptD != arguments.end()) {
		vector<string>::iterator itArgD = itOptD ;
		++itArgD ;
		if ((itArgD == arguments.end()) || ((*itArgD)[0] == '-')) {
			cerr << "Block dimension argument missing; using default" << blockDim << ".\n" ;
		}
		else {
			int d ;
			istringstream is(*itArgD) ;
			is >> d ;
			if (!is)
				cerr << "Bad specification of block dimension argument: " << *itArgD << "; using default " << blockDim << ".\n" ;
			else {
				if (!((d == 2) || (d == 3)))
					cerr << "Invalid block dimension argument: " << d << "; using default " << blockDim << ".\n" ;
				else
					blockDim = d ;
			}
			arguments.erase(itArgD) ;
			itOptD = find(arguments.begin(), arguments.end(), "-d") ;
		}		
		arguments.erase(itOptD) ;
	}
	
	vector<string>::iterator itOptN = find(arguments.begin(), arguments.end(), "-n") ;
	if (itOptN != arguments.end()) {
		vector<string>::iterator itArgN = itOptN ;
		++itArgN ;
		if ((itArgN == arguments.end()) || ((*itArgN)[0] == '-')) {
			cerr << "Maximum number of extracted blocks argument missing; using default " << maxBlocks << ".\n" ;
		}
		else {
			int n ;
			istringstream is(*itArgN) ;
			is >> n ;
			if (!is)
				cerr << "Bad specification of maximum number of extracted blocks argument: " << *itArgN << "; using default " << maxBlocks << ".\n" ;
			else {
				if (!(n > 0))
					cerr << "Invalid maximum number of extracted blocks argument: " << n << "; using default " << maxBlocks << ".\n" ;
				else
					maxBlocks = n ;
			}
			arguments.erase(itArgN) ;
			itOptN = find(arguments.begin(), arguments.end(), "-n") ;
		}		
		arguments.erase(itOptN) ;
	}
	
	vector<string>::iterator itOpt8 = find(arguments.begin(), arguments.end(), "-8") ;
	if (itOpt8 != arguments.end()) {
		force8bitOutput = true ;
		arguments.erase(itOpt8) ;
	}		
	
	vector<string>::iterator itOptM = find(arguments.begin(), arguments.end(), "-m") ;
	if (itOptM != arguments.end()) {
		vector<string>::iterator itArgM = itOptM ;
		++itArgM ;
		int argumentCounter = 0 ; 
		while ((itArgM != arguments.end()) && ((*itArgM)[0] != '-')) {
			mapFilenames.push_back(*itArgM) ;
			++itArgM ;
			++argumentCounter ;
		}
		if (!mapFilenames.size()) {
			cerr << "No segmentations were provided.\n" ;
		}
		else {
			itArgM = itOptM ;
			++itArgM ;
			arguments.erase(itArgM, itArgM + argumentCounter) ;
			itOptM = find(arguments.begin(), arguments.end(), "-m") ;
			arguments.erase(itOptM) ;
            mapFilenamesSpecified = true ;
		}
	}	
	
	vector<string>::iterator itOptT = find(arguments.begin(), arguments.end(), "-t") ;
	if (itOptT != arguments.end()) {
        if (!mapFilenamesSpecified) {
            cerr << "Probability map threshold option specification without specifying probability map filenames; discarding.\n" ;
        }
        else {
            vector<string>::iterator itArgT = itOptT ;
            ++itArgT ;
            if ((itArgT == arguments.end()) || ((*itArgT)[0] == '-')) {
                cerr << "Probability map threshold argument missing, using default " << threshold << ".\n" ;
            }
            else {
                double t ;
                istringstream is(*itArgT) ;
                is >> t ;
                if (!is)
                    cerr << "Bad specification of probability map threshold argument: " << *itArgT << "; using default " << threshold << ".\n" ;
                else {
                    if (!((t > 0) && (t <= 1.0)))
                        cerr << "Invalid probability map threshold argument: " << t << "; using default " << threshold << ".\n" ;
                    else
                        threshold = t ;
                }
                arguments.erase(itArgT) ;
                itOptT = find(arguments.begin(), arguments.end(), "-t") ;
            }
        }
		arguments.erase(itOptT) ;
	}
	
	vector<string>::iterator itOptR = find(arguments.begin(), arguments.end(), "-r") ;
	if (itOptR != arguments.end()) {
        if (!mapFilenamesSpecified) { 
            cerr << "Specification of testing set to training set ratio without specifying probability map filenames; discarding.\n" ;
        }
        else {
            vector<string>::iterator itArgR = itOptR ;
            ++itArgR ;
            if ((itArgR == arguments.end()) || ((*itArgR)[0] == '-')) {
                cerr << "Testing set to training set ratio argument missing, using default " << testTrainRatio << ".\n" ;
            }
            else {
                double r ;
                istringstream is(*itArgR) ;
                is >> r ;
                if (!is)
                    cerr << "Bad specification of testing set to training set ratio argument: " << *itArgR << "; using default " << testTrainRatio << ".\n" ;
                else {
                    if (!((r > 0) && (r <= 1.0)))
                        cerr << "Invalid testing set to training set ratio argument: " << r << "; using default " << testTrainRatio << ".\n" ;
                    else
                        testTrainRatio = r ;
                }
                arguments.erase(itArgR) ;
                itOptR = find(arguments.begin(), arguments.end(), "-r") ;
            }
        }
		arguments.erase(itOptR) ;
	}
	
	vector<string>::iterator itOptO = find(arguments.begin(), arguments.end(), "-o") ;
	if (itOptO != arguments.end()) {
		vector<string>::iterator itArgO = itOptO ;
		++itArgO ;
		if ((itArgO == arguments.end()) || ((*itArgO)[0] == '-')) {
			cerr << "Output prefix argument missing; using input image filename.\n" ;
		}
		else {
			outputPrefix = *itArgO ;
			arguments.erase(itArgO) ;
			itOptO = find(arguments.begin(), arguments.end(), "-o") ;
		}		
		arguments.erase(itOptO) ;
	}	
	
	if (arguments.size()) {
		cerr << "Spurious options/arguments: " ;
		for (int i = 0 ; i < arguments.size() ; ++i)
			cerr << arguments[i] ;
		cerr << '\n' ;
	}
	
	
	//////////////////// PARSE INPUT FILE NAME AND PREPARE EXECUTION ENVIRONMENT AND PARAMETERS

	// Perfom format checking before loading input image
	itk::NiftiImageIO::Pointer niftiIO = itk::NiftiImageIO::New() ;	
	if (!niftiIO->CanReadFile(inputFilename.c_str())) {
		cerr << inputFilename << " is not a valid NII file; terminating.\n" ;
		return 5 ;
	}	
	
	if (!outputPrefix.size()) {
        int lastSlashPos = inputFilename.find_last_of('/') ;
        outputPrefix = inputFilename.substr(lastSlashPos + 1, (inputFilename.length() - 7) - lastSlashPos - 1) ;        
    }
	
	srandom(time(0)) ;	
	
    
	//////////////////// LOAD INPUT IMAGE AND SAVE ITS PROPERTIES
	
 	typedef itk::Image<float,3>                    ImageType ;
	typedef itk::ImageFileReader<ImageType>        ImageReaderType ;
 
	// Read the input image ...
	ImageReaderType::Pointer inputImageReader = ImageReaderType::New() ;
	inputImageReader->SetFileName(inputFilename) ;
 	inputImageReader->SetImageIO(niftiIO) ;
	inputImageReader->Update() ;
 
	ImageType::Pointer inputImage = ImageType::New() ;
    if (force8bitOutput) {
        typedef itk::RescaleIntensityImageFilter<ImageType,ImageType> RescaleFilterType ;
        RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New() ;
        rescaleFilter->SetInput(inputImageReader->GetOutput()) ;
        rescaleFilter->SetOutputMinimum(0) ;
        rescaleFilter->SetOutputMaximum(255) ;
        rescaleFilter->Update() ;
        inputImage->Graft(rescaleFilter->GetOutput()) ;
    }
    else {
        inputImage->Graft(inputImageReader->GetOutput()) ;
    }
    
	// ... save some meta information ...
	typedef itk::ImageBase<3>::RegionType::SizeType SizeType ;
	ImageType::RegionType inputImageRegion = inputImage->GetLargestPossibleRegion() ;
	SizeType imageDimensions = inputImageRegion.GetSize() ;
	
	typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType ;
	StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New() ;
	statisticsImageFilter->SetInput(inputImage) ;
	statisticsImageFilter->Update() ;	
	ImageType::PixelType inputImageMinimum = statisticsImageFilter->GetMinimum() ;
	statisticsImageFilter = 0 ;
    
    if (fillValue == 0)
        fillValue = inputImageMinimum ;
    
    int maxBlockPoints2D = blockSize * blockSize ;
    int maxBlockPoints3D = blockSize * blockSize * blockSize ;
    
    if (!mapFilenamesSpecified) {
        
        ofstream outputFile(string(outputPrefix + ".csv").c_str()) ;
        
        for (int i = 0 ; i < segmentationFilenames.size() ; ++i) {
            
            string segmentationFilename = segmentationFilenames[i] ;
            
            itk::NiftiImageIO::Pointer niftiIO = itk::NiftiImageIO::New() ;		
            if (!niftiIO->CanReadFile(segmentationFilename.c_str())) {
                cerr << segmentationFilename << " is not a valid NII file; terminating.\n" ;
                return 6 ;
            }	

            ImageReaderType::Pointer segmentationReader = ImageReaderType::New() ;
            segmentationReader->SetImageIO(niftiIO) ;
            segmentationReader->SetFileName(segmentationFilename) ;
            segmentationReader->Update() ;
            
            ImageType::Pointer segmentation = ImageType::New() ;
            segmentation->Graft(segmentationReader->GetOutput()) ;
            
            int lastUnderscorePos = segmentationFilename.find_last_of('_') ;
            int nextToLastUnderscorePos = segmentationFilename.substr(0, lastUnderscorePos).find_last_of('_') ;
            string radLexId = segmentationFilename.substr(nextToLastUnderscorePos + 1, lastUnderscorePos - nextToLastUnderscorePos - 1) ;        
            
            // No checking that input image and segmentation images match; the user is supposed to make correct use
            
            vector<int> organVoxelIds ;
            
            ImageType::IndexType index ;
            for (int z = 0 ; z < imageDimensions[2] ; ++z)
                for (int y = 0 ; y < imageDimensions[1] ; ++y)
                    for (int x = 0 ; x < imageDimensions[0] ; ++x) {
                        index[0] = x ; 
                        index[1] = y ; 
                        index[2] = z ;
                        if (segmentation->GetPixel(index) != 0)
                            organVoxelIds.push_back(segmentation->ComputeOffset(index)) ;
                    }
                    
            for (int i = 0 ; i < maxBlocks ; ++i) {
                int voxelId = organVoxelIds[random() % organVoxelIds.size()] ;
                ImageType::IndexType voxelIndex = segmentation->ComputeIndex(voxelId) ;
                
                outputFile << radLexId << ',' ;
                
                ImageType::PixelType intensity ;
                
                if (blockDim == 2) {
                    int j = 0 ;
                    for (int y = voxelIndex[1] - blockSize/2 ; y <= voxelIndex[1] + blockSize/2 ; ++y)
                        for (int x = voxelIndex[0] - blockSize/2 ; x <= voxelIndex[0] + blockSize/2 ; ++x) {
                            index[0] = x ;
                            index[1] = y ;
                            index[2] = voxelIndex[2] ;                         
                            if (inputImageRegion.IsInside(index))
                                intensity = inputImage->GetPixel(index) ;
                            else
                                intensity = fillValue ;
                            outputFile << intensity ;
                            if ((j+1) != maxBlockPoints2D)
                                outputFile << ',' ;
                            ++j ;
                        }
                }
                else {
                    int j = 0 ;
                    for (int z = voxelIndex[2] - blockSize/2 ; z <= voxelIndex[2] + blockSize/2 ; ++z)
                        for (int y = voxelIndex[1] - blockSize/2 ; y <= voxelIndex[1] + blockSize/2 ; ++y)
                            for (int x = voxelIndex[0] - blockSize/2 ; x <= voxelIndex[0] + blockSize/2 ; ++x) {
                                index[0] = x ;
                                index[1] = y ;
                                index[2] = z ;                         
                                if (inputImageRegion.IsInside(index))
                                    intensity = inputImage->GetPixel(index) ;
                                else
                                    intensity = fillValue ;
                                outputFile << intensity ;
                                if ((j+1) != maxBlockPoints3D)
                                    outputFile << ',' ;
                                ++j ;
                            }
                }
                        
                outputFile << '\n' ;
                
                vector<int>::iterator it = find(organVoxelIds.begin(), organVoxelIds.end(), voxelId) ;
                organVoxelIds.erase(it) ;
                
                if (!organVoxelIds.size() && (i < (maxBlocks - 1))) {
                    cout << "Not enough voxels in " << segmentationFilename << "; using all available.\n" ;
                    break ;
                }                    
            }
        }	
    }
    else {
        ofstream testOutputFile(string(outputPrefix + "-test.csv").c_str()) ;
        ofstream trainOutputFile(string(outputPrefix + "-train.csv").c_str()) ;
        
        for (int i = 0 ; i < mapFilenames.size() ; ++i) {
            
            string mapFilename = mapFilenames[i] ;
            
            itk::MetaImageIO::Pointer metaIO = itk::MetaImageIO::New() ;		
            if (!metaIO->CanReadFile(mapFilename.c_str())) {
                cout << mapFilename << " is not a valid Meta Image file; terminating.\n" ;
                return 7 ;
            }
            
            int lastSlashPos = mapFilename.find_last_of('/') ;
            int lastDotPos = mapFilename.find_last_of('.') ;
            string auxMapFilename = mapFilename.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) ;
            
            istringstream iss(auxMapFilename);
            string token;
            getline(iss, token, '-') ;
            getline(iss, token, '-') ;
            string radLexId = token ;            
            
            string segmentationFilename ;
            for (int j = 0 ; j < segmentationFilenames.size() ; ++j) 
                if (segmentationFilenames[j].find(radLexId) != string::npos) {
                    segmentationFilename = segmentationFilenames[j] ;
                    break ;
                }
                
         	if (!segmentationFilename.size()) {
                cout << mapFilename << " has no corresponding segmentation file; proceeding with next map file.\n" ;
                continue ;
            }

            ImageReaderType::Pointer mapFileReader = ImageReaderType::New() ;
            mapFileReader->SetImageIO(metaIO) ;
            mapFileReader->SetFileName(mapFilename) ;
            mapFileReader->Update() ;
            
            ImageType::Pointer mapImage = ImageType::New() ;
            mapImage->Graft(mapFileReader->GetOutput()) ;            
            
            StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New() ;
            statisticsImageFilter->SetInput(mapImage) ;
            statisticsImageFilter->Update() ;	
            ImageType::PixelType mapMaximum = statisticsImageFilter->GetMaximum() ;	
            statisticsImageFilter = 0 ;
            
            itk::NiftiImageIO::Pointer niftiIO = itk::NiftiImageIO::New() ;		
            if (!niftiIO->CanReadFile(segmentationFilename.c_str())) {
                cerr << segmentationFilename << " is not a valid NII file; terminating.\n" ;
                return 8 ;
            }	

            ImageReaderType::Pointer segmentationReader = ImageReaderType::New() ;
            segmentationReader->SetImageIO(niftiIO) ;
            segmentationReader->SetFileName(segmentationFilename) ;
            segmentationReader->Update() ;
            
            ImageType::Pointer segmentation = ImageType::New() ;
            segmentation->Graft(segmentationReader->GetOutput()) ;            
            
            vector<int> organVoxelIds ;
            vector<int> mapRegionVoxelIds ;
            
            double thresh = mapMaximum * threshold ;            
            ImageType::IndexType index ;
            
            for (int z = 0 ; z < imageDimensions[2] ; ++z)
                for (int y = 0 ; y < imageDimensions[1] ; ++y)
                    for (int x = 0 ; x < imageDimensions[0] ; ++x) {
                        index[0] = x ; 
                        index[1] = y ; 
                        index[2] = z ;
                        if (segmentation->GetPixel(index) != 0)
                            organVoxelIds.push_back(segmentation->ComputeOffset(index)) ;
                    }            
            
            for (int z = 0 ; z < imageDimensions[2] ; ++z)
                for (int y = 0 ; y < imageDimensions[1] ; ++y)
                    for (int x = 0 ; x < imageDimensions[0] ; ++x) {
                        index[0] = x ;
                        index[1] = y ;
                        index[2] = z ;                        
                        ImageType::PixelType prob = mapImage->GetPixel(index) ;                        
                        if (prob >= thresh)
                            mapRegionVoxelIds.push_back(mapImage->ComputeOffset(index)) ;
                    }                    
            
            set<int> organVoxelIdsSet(organVoxelIds.begin(), organVoxelIds.end()) ;
            set<int> mapRegionVoxelIdsSet(mapRegionVoxelIds.begin(), mapRegionVoxelIds.end()) ;
        
            organVoxelIds.clear() ;
            set_difference(organVoxelIdsSet.begin(), organVoxelIdsSet.end(), mapRegionVoxelIdsSet.begin(), mapRegionVoxelIdsSet.end(), back_inserter(organVoxelIds)) ;
                
            int maxTestingBlocks = testTrainRatio * maxBlocks ;
            int maxTrainingBlocks = maxBlocks - maxTestingBlocks ;         

            for (int j = 0 ; j < maxTrainingBlocks ; ++j) {
                int voxelId = mapRegionVoxelIds[random() % mapRegionVoxelIds.size()] ;
                ImageType::IndexType voxelIndex = mapImage->ComputeIndex(voxelId) ;
                
                trainOutputFile << radLexId << ',' ;
                
                ImageType::PixelType intensity ;
                
                if (blockDim == 2) {
                    int k = 0 ;
                    for (int y = voxelIndex[1] - blockSize/2 ; y <= voxelIndex[1] + blockSize/2 ; ++y)
                        for (int x = voxelIndex[0] - blockSize/2 ; x <= voxelIndex[0] + blockSize/2 ; ++x) {
                            index[0] = x ;
                            index[1] = y ;
                            index[2] = voxelIndex[2] ;                         
                            if (inputImageRegion.IsInside(index))
                                intensity = inputImage->GetPixel(index) ;
                            else
                                intensity = fillValue ;
                            trainOutputFile << intensity ;
                            if ((k+1) != maxBlockPoints2D)
                                trainOutputFile << ',' ;
                            ++k ;
                        }
                }
                else {
                    int k = 0 ;
                    for (int z = voxelIndex[2] - blockSize/2 ; z <= voxelIndex[2] + blockSize/2 ; ++z)
                        for (int y = voxelIndex[1] - blockSize/2 ; y <= voxelIndex[1] + blockSize/2 ; ++y)
                            for (int x = voxelIndex[0] - blockSize/2 ; x <= voxelIndex[0] + blockSize/2 ; ++x) {
                                index[0] = x ;
                                index[1] = y ;
                                index[2] = z ;                         
                                if (inputImageRegion.IsInside(index))
                                    intensity = inputImage->GetPixel(index) ;
                                else
                                    intensity = fillValue ;
                                trainOutputFile << intensity ;
                                if ((k+1) != maxBlockPoints3D)
                                    trainOutputFile << ',' ;
                                ++k ;
                            }
                }
                        
                trainOutputFile << '\n' ;
                
                vector<int>::iterator it = find(mapRegionVoxelIds.begin(), mapRegionVoxelIds.end(), voxelId) ;
                mapRegionVoxelIds.erase(it) ;
                
                if (!mapRegionVoxelIds.size() && (i < (maxTrainingBlocks - 1))) {
                    cout << "Not enough voxels in " << mapFilename << " region; using all available.\n" ;
                    break ;
                }                    
            }

            for (int j = 0 ; j < maxTestingBlocks ; ++j) {
                int voxelId = organVoxelIds[random() % organVoxelIds.size()] ;
                ImageType::IndexType voxelIndex = segmentation->ComputeIndex(voxelId) ;
                
                testOutputFile << radLexId << ',' ;
                
                ImageType::PixelType intensity ;
                
                if (blockDim == 2) {
                    int k = 0 ;
                    for (int y = voxelIndex[1] - blockSize/2 ; y <= voxelIndex[1] + blockSize/2 ; ++y)
                        for (int x = voxelIndex[0] - blockSize/2 ; x <= voxelIndex[0] + blockSize/2 ; ++x) {
                            index[0] = x ;
                            index[1] = y ;
                            index[2] = voxelIndex[2] ;                         
                            if (inputImageRegion.IsInside(index))
                                intensity = inputImage->GetPixel(index) ;
                            else
                                intensity = fillValue ;
                            testOutputFile << intensity ;
                            if ((k+1) != maxBlockPoints2D)
                                testOutputFile << ',' ;
                            ++k ;
                        }
                }
                else {
                    int k = 0 ;
                    for (int z = voxelIndex[2] - blockSize/2 ; z <= voxelIndex[2] + blockSize/2 ; ++z)
                        for (int y = voxelIndex[1] - blockSize/2 ; y <= voxelIndex[1] + blockSize/2 ; ++y)
                            for (int x = voxelIndex[0] - blockSize/2 ; x <= voxelIndex[0] + blockSize/2 ; ++x) {
                                index[0] = x ;
                                index[1] = y ;
                                index[2] = z ;                         
                                if (inputImageRegion.IsInside(index))
                                    intensity = inputImage->GetPixel(index) ;
                                else
                                    intensity = fillValue ;
                                testOutputFile << intensity ;
                                if ((k+1) != maxBlockPoints3D)
                                    testOutputFile << ',' ;
                                ++k ;
                            }
                }
                        
                testOutputFile << '\n' ;
                
                vector<int>::iterator it = find(organVoxelIds.begin(), organVoxelIds.end(), voxelId) ;
                organVoxelIds.erase(it) ;
                
                if (!organVoxelIds.size() && (i < (maxTestingBlocks - 1))) {
                    cout << "Not enough voxels in " << segmentationFilename << "; using all available.\n" ;
                    break ;
                }                    
            }            
        }
    }
	
    inputImage = 0 ;	
	inputImageReader = 0 ;	
    niftiIO = 0 ;
}
