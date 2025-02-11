#include <iostream>

using namespace std;
void ShowDeviceInfor()
{
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << "Information GPU Card: " << endl;
    cout << "1/GPU card's name: " << prop.name << endl;
    cout << "2/GPU computation capabilities: " << prop.major << "." << prop.minor << endl;
    cout << "3/Maximum number of block dimensions: " 
            << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
    cout << "4/Maximum number of grid dimensions: " 
            << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    cout << "5/Maximum size of GPU memory: " << prop.totalGlobalMem << " bytes" << endl;
    cout<<"6/Amount of constant and share memory: "<<endl;
    cout << "Amount of constant memory: " << prop.totalConstMem << " bytes" << endl;
    cout << "Amount of shared memory: " << prop.sharedMemPerBlock << " bytes" << endl;
    cout << "7/Warp size: " << prop.warpSize << endl;

}
int main(int argc, char ** argv) {
    ShowDeviceInfor();
    return 0;
}