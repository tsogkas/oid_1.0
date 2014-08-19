#include "mex.h"
#include <math.h>

static inline int min(int x, int y) { return (x <= y ? x : y); }

mxArray *process(const mxArray *mx_words, 
                 const mxArray *mx_num_words, 
                 const mxArray *mx_sbin) 
{
  double *words = (double *)mxGetPr(mx_words);
  const int *dims = mxGetDimensions(mx_words);

  int sbin = (int)mxGetScalar(mx_sbin);
  int num_words = (int)mxGetScalar(mx_num_words);

  int blocks[3];
  blocks[0] = (int)round((double)dims[0]/(double)sbin);
  blocks[1] = (int)round((double)dims[1]/(double)sbin);
  blocks[2] = num_words;
  mxArray *mx_hist = mxCreateNumericArray(3, blocks, mxSINGLE_CLASS, mxREAL);
  float *hist = (float *)mxGetPr(mx_hist);

  for (int x = 0; x < dims[1]; x++) {
    for (int y = 0; y < dims[0]; y++) {
      int word = *(words + dims[0]*x + y) - 1;

      int bx = min(x/sbin, blocks[1]-1);
      int by = min(y/sbin, blocks[0]-1);

      *(hist + by + blocks[0]*bx + blocks[0]*blocks[1]*word) += 1;
    }
  }

  return mx_hist;
}

// matlab entry point
//  hists = color_word_hists(word_image, num_words, bin_size)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 3)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");

  if (mxGetNumberOfDimensions(prhs[0]) != 2 || 
      mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input: word_image");

  plhs[0] = process(prhs[0], prhs[1], prhs[2]);
}



