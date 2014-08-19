#include <math.h>
#include "mex.h"

// small value, used to avoid division by zero
const float eps = 0.0001;

// unit vectors used to compute gradient orientation
const double uu[9] = {
  1.0000, 
  0.9397, 
  0.7660, 
  0.500, 
  0.1736, 
  -0.1736, 
  -0.5000, 
  -0.7660, 
  -0.9397
};

const double vv[9] = {
  0.0000, 
  0.3420, 
  0.6428, 
  0.8660, 
  0.9848, 
  0.9848, 
  0.8660, 
  0.6428, 
  0.3420
};

template <typename T>
static inline T min(T x, T y) { return (x <= y ? x : y); }

template <typename T>
static inline T max(T x, T y) { return (x <= y ? y : x); }

// main function:
// takes a double color image and a HOG cell size
// returns HOG features
mxArray *process(const mxArray *mx_image, const mxArray *mx_cell_size) {
  double *im = (double *)mxGetPr(mx_image);
  const int *dims = mxGetDimensions(mx_image);
  if (mxGetNumberOfDimensions(mx_image) != 3 || dims[2] != 3 ||
      mxGetClassID(mx_image) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input (not color or not class double)");

  int cell_size = (int)mxGetScalar(mx_cell_size);
  int cells[3];
  cells[0] = (int)round((double)dims[0]/(double)cell_size);
  cells[1] = (int)round((double)dims[1]/(double)cell_size);
  cells[2] = 9+18+4+1;
  int cells_h_x_w = cells[0]*cells[1];

  // memory for caching orientation histograms & their norms
  float *hist = (float *)mxCalloc(cells_h_x_w*18, sizeof(float));
  float *norm = (float *)mxCalloc(cells_h_x_w, sizeof(float));

  // memory for HOG features
  mxArray *mx_feat = mxCreateNumericArray(3, cells, mxSINGLE_CLASS, mxREAL);
  float *feat = (float *)mxGetPr(mx_feat);
  
  { // compute oriented gradient histograms
    int visible[2];
    visible[0] = cells[0]*cell_size;
    visible[1] = cells[1]*cell_size;
    
    for (int x = 1; x < visible[1]-1; x++) {
      for (int y = 1; y < visible[0]-1; y++) {
        // first color channel
        double *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
        double dy = *(s+1) - *(s-1);
        double dx = *(s+dims[0]) - *(s-dims[0]);
        double v = dx*dx + dy*dy;

        // second color channel
        s += dims[0]*dims[1];
        double dy2 = *(s+1) - *(s-1);
        double dx2 = *(s+dims[0]) - *(s-dims[0]);
        double v2 = dx2*dx2 + dy2*dy2;

        // third color channel
        s += dims[0]*dims[1];
        double dy3 = *(s+1) - *(s-1);
        double dx3 = *(s+dims[0]) - *(s-dims[0]);
        double v3 = dx3*dx3 + dy3*dy3;

        // pick channel with strongest gradient
        if (v2 > v) {
          v = v2;
          dx = dx2;
          dy = dy2;
        } 
        if (v3 > v) {
          v = v3;
          dx = dx3;
          dy = dy3;
        }

        // snap to one of 18 orientations
        double best_dot = 0;
        int best_o = 0;
        for (int o = 0; o < 9; o++) {
          double dot = uu[o]*dx + vv[o]*dy;
          if (dot > best_dot) {
            best_dot = dot;
            best_o = o;
          } else if (-dot > best_dot) {
            best_dot = -dot;
            best_o = o+9;
          }
        }
        
        // add to 4 histograms around pixel using linear interpolation
        double xp = ((double)x+0.5)/(double)cell_size - 0.5;
        double yp = ((double)y+0.5)/(double)cell_size - 0.5;
        int ixp = (int)floor(xp);
        int iyp = (int)floor(yp);
        double vx0 = xp-ixp;
        double vy0 = yp-iyp;
        double vx1 = 1.0-vx0;
        double vy1 = 1.0-vy0;
        v = sqrt(v);

        if (ixp >= 0 && iyp >= 0) {
          *(hist + ixp*cells[0] + iyp + best_o*cells_h_x_w) += 
            vx1*vy1*v;
        }

        if (ixp+1 < cells[1] && iyp >= 0) {
          *(hist + (ixp+1)*cells[0] + iyp + best_o*cells_h_x_w) += 
            vx0*vy1*v;
        }

        if (ixp >= 0 && iyp+1 < cells[0]) {
          *(hist + ixp*cells[0] + (iyp+1) + best_o*cells_h_x_w) += 
            vx1*vy0*v;
        }

        if (ixp+1 < cells[1] && iyp+1 < cells[0]) {
          *(hist + (ixp+1)*cells[0] + (iyp+1) + best_o*cells_h_x_w) += 
            vx0*vy0*v;
        }
      }
    }
  }

  { // compute energy in each cell by summing over orientations
    for (int o = 0; o < 9; o++) {
      float *src1 = hist + o*cells_h_x_w;
      float *src2 = hist + (o+9)*cells_h_x_w;
      float *dst = norm;
      float *end = norm + cells_h_x_w;
      while (dst < end) {
        *(dst++) += (*src1 + *src2) * (*src1 + *src2);
        src1++;
        src2++;
      }
    }
  }

  { // compute features
    // normalization factors needed to make analytic projections L2 unit norm
    const float sqrt_18_inv = 1.0 / sqrt(18);
    const float sqrt_4_inv  = 0.5;

    for (int x = 0; x < cells[1]; x++) {
      // xm1|x|xp1
      int xm1 = max(x-1, 0);
      int xp1 = min(x+1, cells[1]-1);

      for (int y = 0; y < cells[0]; y++) {
        float *dst = feat + x*cells[0] + y;      
        float *src, n1, n2, n3, n4;
        float *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9;

        //   ym1
        //   ---
        //    y
        //   ---
        //   yp1
        int ym1 = max(y-1, 0);
        int yp1 = min(y+1, cells[0]-1);

        // Cells around p5 = (x,y) used for normalization
        //
        // p1|p2|p3
        // --+--+--
        // p4|p5|p6
        // --+--+--
        // p7|p8|p9
        p1 = norm + xm1*cells[0] + ym1;
        p2 = norm + x  *cells[0] + ym1;
        p3 = norm + xp1*cells[0] + ym1;
        p4 = norm + xm1*cells[0] + y  ;
        p5 = norm + x  *cells[0] + y  ;
        p6 = norm + xp1*cells[0] + y  ;
        p7 = norm + xm1*cells[0] + yp1;
        p8 = norm + x  *cells[0] + yp1;
        p9 = norm + xp1*cells[0] + yp1;

        // Normalization blocks (4 2x2 cell blocks around p5 = (x,y))
        // n1:         n2:         n3:         n4:
        //    p5|p6       p2|p2       p4|p5       p1|p2
        //    --+--       --+--       --+--       --+--
        //    p8|p9       p5|p6       p7|p8       p4|p5
        n1 = 1.0 / sqrt(*p5 + *p6 + *p8 + *p9 + eps);
        n2 = 1.0 / sqrt(*p2 + *p3 + *p5 + *p6 + eps);
        n3 = 1.0 / sqrt(*p4 + *p5 + *p7 + *p8 + eps);
        n4 = 1.0 / sqrt(*p1 + *p2 + *p4 + *p5 + eps);

        // texture energy features
        float t1 = 0;
        float t2 = 0;
        float t3 = 0;
        float t4 = 0;

        // contrast-sensitive features
        src = hist + x*cells[0] + y;
        for (int o = 0; o < 18; o++) {
          float h1 = min(*src * n1, 0.2f);
          float h2 = min(*src * n2, 0.2f);
          float h3 = min(*src * n3, 0.2f);
          float h4 = min(*src * n4, 0.2f);
          *dst = sqrt_4_inv * (h1 + h2 + h3 + h4);
          t1 += h1;
          t2 += h2;
          t3 += h3;
          t4 += h4;
          dst += cells_h_x_w;
          src += cells_h_x_w;
        }

        // contrast-insensitive features
        src = hist + x*cells[0] + y;
        for (int o = 0; o < 9; o++) {
          float sum = *src + *(src + 9*cells_h_x_w);
          float h1 = min(sum * n1, 0.2f);
          float h2 = min(sum * n2, 0.2f);
          float h3 = min(sum * n3, 0.2f);
          float h4 = min(sum * n4, 0.2f);
          *dst = sqrt_4_inv * (h1 + h2 + h3 + h4);
          dst += cells_h_x_w;
          src += cells_h_x_w;
        }

        // normalize texture energy features
        *dst = sqrt_18_inv * t1;
        dst += cells_h_x_w;
        *dst = sqrt_18_inv * t2;
        dst += cells_h_x_w;
        *dst = sqrt_18_inv * t3;
        dst += cells_h_x_w;
        *dst = sqrt_18_inv * t4;

        // truncation feature
        dst += cells_h_x_w;
        *dst = 0;
      }
    }
  }

  mxFree(hist);
  mxFree(norm);
  return mx_feat;
}

// matlab entry point
// F = features(image, cell_size)
// image should be color with double values
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 2)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");
  plhs[0] = process(prhs[0], prhs[1]);
}
