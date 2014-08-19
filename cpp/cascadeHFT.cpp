#include <xmmintrin.h>
#include <stdint.h>
#include <cassert>
#include <limits>
#include <cmath>
#include <algorithm>
#include "mex.h"

// OS X aligns all memory at 16-byte boundaries (and doesn't provide
// memalign/posix_memalign).  On linux, we use memalign to allocated
// 16-byte aligned memory.
#if !defined(__APPLE__)
#include <malloc.h>
#define malloc_aligned(a,b) memalign(a,b)
#else
#define malloc_aligned(a,b) malloc(b)
#endif

#define IS_ALIGNED(ptr) ((((uintptr_t)(ptr)) & 0xF) == 0)

// N.B. If you change the number of features you will need to unroll
// the unrolled loop in process() more.
#define NUM_FEATURES 32

struct thread_data {
    float *A;
    float *B;
    float *C;
    int *pos;
    int nPos;
    const mwSize *A_dims;
    const mwSize *B_dims;
};


float *align(const int *dims) {
    float *F = (float *)malloc_aligned(16, dims[0]*dims[1]*NUM_FEATURES*sizeof(float));
    // Sanity check that memory is aligned
    if (!IS_ALIGNED(F))
        mexErrMsgTxt("Memory not aligned");
    return F;
}

void initializeAlignedMemory(float *p, const float *in, const int *dims) {
    for (int x = 0; x < dims[1]; ++x)
        for (int y = 0; y < dims[0]; ++y) {
            for (int f = 0; f < dims[2]; ++f)
                *(p++) = in[y + f*dims[0]*dims[1] + x*dims[0]];
            for (int f = dims[2]; f < NUM_FEATURES; ++f)
                *(p++) = 0;
        }
}

// --- Compute feature variance --------------------------------------------------
float *variance(const float *in, const int *dims) {
    float *var = static_cast<float *>(mxMalloc(dims[0]*dims[1]*sizeof(float)));
    float mean,squareSum;
    for (int x=0; x<dims[1]; ++x)
        for (int y=0; y<dims[0]; ++y) {
            mean = 0;
            for (int f=0; f<dims[2]; ++f)
                mean += in[y+x*dims[0]+f*dims[0]*dims[1]];
            mean /= dims[2];
            squareSum = 0;
            for (int f=0; f<dims[2]; ++f) {
                float tmp = in[y+x*dims[0]+f*dims[0]*dims[1]];
                squareSum += (tmp - mean)*(tmp - mean);
            }
            var[y+x*dims[0]] = squareSum / dims[2];
        }
    return var;
}


// --- Optimized convolution of 3D matrix with a 3D filter using SSE instructions
void convSSE(thread_data &args) {
    float *A = args.A;
    float *B = args.B;
    float *C = args.C;
    const mwSize *A_dims = args.A_dims;
    const mwSize *B_dims = args.B_dims;
    int *pos = args.pos;
    int nPos = args.nPos;

    __m128 a,b,c;
    float *dst = C;
    for (int p = 0; p < nPos; ++p) {
        int x = pos[p] / A_dims[0];	// we suppose that we already have
        int y = pos[p] % A_dims[0];	// C++ indexing for pos
        __m128 v = _mm_setzero_ps();
        const float *A_src = A + y*NUM_FEATURES + x*A_dims[0]*NUM_FEATURES;
        const float *B_src = B;
        for (int xp = 0; xp < B_dims[1]; ++xp) {
            const float *A_off = A_src;
            const float *B_off = B_src;
            for (int yp = 0; yp < B_dims[0]; ++yp) {
                a = _mm_load_ps(A_off+0);
                b = _mm_load_ps(B_off+0);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                a = _mm_load_ps(A_off+4);
                b = _mm_load_ps(B_off+4);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                a = _mm_load_ps(A_off+8);
                b = _mm_load_ps(B_off+8);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                a = _mm_load_ps(A_off+12);
                b = _mm_load_ps(B_off+12);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                a = _mm_load_ps(A_off+16);
                b = _mm_load_ps(B_off+16);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                a = _mm_load_ps(A_off+20);
                b = _mm_load_ps(B_off+20);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                a = _mm_load_ps(A_off+24);
                b = _mm_load_ps(B_off+24);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                a = _mm_load_ps(A_off+28);
                b = _mm_load_ps(B_off+28);
                c = _mm_mul_ps(a, b);
                v = _mm_add_ps(v, c);

                // N.B. Unroll me more/less if you change NUM_FEATURES

                A_off += NUM_FEATURES;
                B_off += NUM_FEATURES;
            }

            A_src += A_dims[0]*NUM_FEATURES;
            B_src += B_dims[0]*NUM_FEATURES;
        }
        // buf[] must be 16-byte aligned
        float buf[4] __attribute__ ((aligned (16)));
        _mm_store_ps(buf, v);
        _mm_empty();
        *(dst + pos[p]) = buf[0]+buf[1]+buf[2]+buf[3];	// statsogk
    }
}

// --- Typical convolution of 2D matrix with a 2D filter -----------------------
void conv(thread_data &args, float pe) {
    float *A = args.A;
    float *B = args.B;
    float *C = args.C;
    const mwSize *A_dims = args.A_dims;
    const mwSize *B_dims = args.B_dims;
    int *pos = args.pos;
    int nPos = args.nPos;

    float *dst = C;
    float *A_src = A;
    float *B_src = B;
    for (int p = 0; p < nPos; ++p) {
        int x = pos[p] / A_dims[0];
        int y = pos[p] % A_dims[0];
        // double val = 0;
        float val = 0;
        for (int xp = 0; xp < B_dims[1]; ++xp) {
            float *A_off = A_src + (x+xp)*A_dims[0] + y;
            float *B_off = B_src + xp*B_dims[0];
            switch(B_dims[0]) {
            case 20: val += A_off[19] * B_off[19];
            case 19: val += A_off[18] * B_off[18];
            case 18: val += A_off[17] * B_off[17];
            case 17: val += A_off[16] * B_off[16];
            case 16: val += A_off[15] * B_off[15];
            case 15: val += A_off[14] * B_off[14];
            case 14: val += A_off[13] * B_off[13];
            case 13: val += A_off[12] * B_off[12];
            case 12: val += A_off[11] * B_off[11];
            case 11: val += A_off[10] * B_off[10];
            case 10: val += A_off[9] * B_off[9];
            case 9: val += A_off[8] * B_off[8];
            case 8: val += A_off[7] * B_off[7];
            case 7: val += A_off[6] * B_off[6];
            case 6: val += A_off[5] * B_off[5];
            case 5: val += A_off[4] * B_off[4];
            case 4: val += A_off[3] * B_off[3];
            case 3: val += A_off[2] * B_off[2];
            case 2: val += A_off[1] * B_off[1];
            case 1: val += A_off[0] * B_off[0];
                break;
            default:
                for (int yp = 0; yp < B_dims[0]; ++yp)
                    val += *(A_off++) * *(B_off++);
            }
        }
        *(dst + pos[p]) += sqrt(val/pe);
    }
}



// ---- matlab entry point --------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // Read features and opts struct (inputs)
    const mxArray *mxFeat = prhs[0];    // features
    const mxArray *mxOpts = prhs[1];    // struct array with various options/values
    const mwSize *featSize= mxGetDimensions(mxFeat);
    const mxArray *mxTree          = mxGetField(mxOpts,0,"tree");
    const mxArray *cellLevels      = mxGetField(mxOpts,0,"levels");
    const mxArray *cellFilters     = mxGetField(mxOpts,0,"filters");
    const mxArray *cellFiltersFlip = mxGetField(mxOpts,0,"filtersFlip");
    const mxArray *cellVarCell     = mxGetField(mxOpts,0,"varCell");
    const mxArray *cellVarCellFlip = mxGetField(mxOpts,0,"varCellFlip");
    const float *feat       = static_cast<const float *>(mxGetData(mxFeat));
    const float pe          = static_cast<const float>(mxGetScalar(mxGetField(mxOpts,0,"pe")));
    const float rThresh     = static_cast<const float>(mxGetScalar(mxGetField(mxOpts,0,"thresh")));
    const int *rootSize     = static_cast<const int *>(mxGetData(mxGetField(mxOpts,0,"rootSize")));
    const int nLevels       = static_cast<const int>(mxGetScalar(mxGetField(mxOpts,0,"nLevels")));
    const int treeLength    = static_cast<const int>(mxGetScalar(mxGetField(mxOpts,0,"treeLength")));
    const bool *isLeaf      = static_cast<const bool*>(mxGetData(mxGetField(mxOpts,0,"isleaf")));
    const int *nNodes       = static_cast<const int *>(mxGetData(mxGetField(mxOpts,0,"nNodes")));
    const int nPixels       = featSize[0]*featSize[1];
    float *varf             = variance(feat,featSize); // feature variance
    // use empirical threshold for each node
    const int useThreshPerNode = static_cast<const int>(mxGetScalar(mxGetField(mxOpts,0,"nodeThresh")));

    // Create position indices arrays
    const int nMaxPos = (featSize[0]-rootSize[0]+1)*(featSize[1]-rootSize[1]+1);
    int *pos       = static_cast<int *>(mxMalloc(nMaxPos*treeLength*sizeof(int)));   // position indices
    int *posFlip   = static_cast<int *>(mxMalloc(nMaxPos*treeLength*sizeof(int)));   // position indices
    int *nPos      = static_cast<int *>(mxCalloc(treeLength,sizeof(int)));
    int *nPosFlip  = static_cast<int *>(mxCalloc(treeLength,sizeof(int)));
    thread_data td; // convolution arguments

    // Create outputs (upper bounds and exact scores arrays)
    int dims[3]  = {0,0,0};
    dims[0] = featSize[0];
    dims[1] = featSize[1];
    dims[2] = treeLength;
    plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);   // upper bounds
    plhs[1] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    float *ub     = static_cast<float *>(mxGetData(plhs[0]));
    float *ubFlip = static_cast<float *>(mxGetData(plhs[1]));


    // --- Start processing ---------------------------------------------------------
    // ------------------------------------------------------------------------------

    // Align memory for filters and intermediate array only once
    //float *B = align(rootSize);
    float *A = align(featSize);
    initializeAlignedMemory(A,feat,featSize);
    // Initialize output arrays
    float minusInfinity = -std::numeric_limits<float>::infinity();
    for (int i=0, iend = dims[0]*dims[1]*dims[2]; i<iend; ++i) {
        ub[i]     = minusInfinity;
        ubFlip[i] = minusInfinity;
    }
    // Initialize positions for root node
    int *pPos = pos + nMaxPos*(treeLength-1);
    int *pPosFlip = posFlip + nMaxPos*(treeLength-1);
    for (int x=0; x<featSize[1]-rootSize[1]+1; ++x)
        for (int y=0; y<featSize[0]-rootSize[0]+1; ++y) {
            *(pPos++) = *(pPosFlip++) = x*featSize[0] + y;
            ++nPos[treeLength-1];
            ++nPosFlip[treeLength-1];
        }

    // For every level in the hierarchy tree (going from the root towards the leaves)
    for (int iLevel = 0; iLevel < nLevels; ++iLevel) {
        int *nodes = static_cast<int*>(mxGetData(mxGetCell(cellLevels,iLevel)));

        // For every node at the current level
        for (int iNode = 0; iNode < nNodes[iLevel]; iNode ++) {
            int node = nodes[iNode]-1; // !!!! subtract one for C++ indexing (watch this on later usages)
            mxArray *mxFilter = mxGetCell(cellFilters,node);
            const mwSize *filterSize = mxGetDimensions(mxFilter);
            float *filter = static_cast<float *>(mxGetData(mxFilter));

            // If you are not at the first level of the hierarchy, find the
            // convolution positions for each node
            if (iLevel) {
                int parent         = static_cast<int>(mxGetScalar(mxGetField(mxTree,node,"parent")))-1;
                int *parentSize    = static_cast<int *>(mxGetData(mxGetField(mxTree,parent,"shape")));
                int *relativeShift = static_cast<int *>(mxGetData(mxGetField(mxTree,node,"relativeShift")));
                float thresh = 0;
                if (useThreshPerNode) {
                    float eThresh = static_cast<float>(mxGetScalar(mxGetField(mxTree,parent,"thresh")));
                    thresh = std::max(rThresh,eThresh);
                }
                else {
                    thresh = rThresh;
                }
                int outSize[2] = {0,0};
                outSize[0] = featSize[0]-filterSize[0]-1;
                outSize[1] = featSize[1]-filterSize[1]-1;
                float *p = ub + parent*nPixels;
                int *parentPos = pos + nMaxPos*parent;
                int *nodePos   = pos + nMaxPos*node;
                int n = 0, row = 0, col = 0;
                for (int i=0; i<nPos[parent]; ++i)
                    if (p[parentPos[i]] > thresh) {       // upper bound is larger than threshold
                        row = parentPos[i] % featSize[0] + relativeShift[0]; // we suppose that we already use c++ indexing for pos
                        col = parentPos[i] / featSize[0] + relativeShift[1];
                        if ((row < outSize[0]) && (col < outSize[1])) // check if inside bounds
                            nodePos[n++] = col*featSize[0] + row;
                    }
                assert(n <= nPos[parent]);
                nPos[node] = n;
                p = ubFlip + parent*nPixels;
                parentPos = posFlip + nMaxPos*parent;
                nodePos   = posFlip + nMaxPos*node;
                n = 0;
                for (int i=0; i<nPosFlip[parent]; ++i)
                    if (p[parentPos[i]] > thresh) {
                        row = parentPos[i] % featSize[0] + relativeShift[0];
                        col = parentPos[i] / featSize[0] + parentSize[1] - relativeShift[1] - filterSize[1];
                        if ((row < outSize[0]) && (col < outSize[1]))
                            nodePos[n++] = col*featSize[0] + row;
                    }
                assert(n <= nPosFlip[parent]);
                nPosFlip[node] = n;
            }

            // Perform 3d convolutions
            // Initialize filter before convolution
            float *B = align(filterSize);  // is this necessary?
            initializeAlignedMemory(B,filter,filterSize);
            td.A = A;
            td.B = B;
            td.A_dims = featSize;
            td.B_dims = filterSize;
            td.C      = ub + node*nPixels;
            td.pos    = pos + nMaxPos*node;
            td.nPos   = nPos[node];
            convSSE(td);

            // Convolve with flipped filter
            float *filterFlip = static_cast<float *>(mxGetData(mxGetCell(cellFiltersFlip,node)));
            initializeAlignedMemory(B,filterFlip,filterSize);
            td.C    = ubFlip + node*nPixels;
            td.pos  = posFlip + nMaxPos*node;
            td.nPos = nPosFlip[node];
            convSSE(td);

            free(B);
            // If we are not at a leaf node, compute the upper bound (includes
            // convolving the feature variance with the cell error variance)
            if (!isLeaf[node]) {
                const int nLeaves = static_cast<const int>(mxGetScalar(mxGetField(mxTree,node,"nLeaves")));
                td.A = varf;
                td.B = static_cast<float*>(mxGetData(mxGetCell(cellVarCell,node)));
                td.B_dims = mxGetDimensions(mxGetCell(cellVarCell,node));
                td.C    = ub + node*nPixels;
                td.pos  = pos + nMaxPos*node;
                td.nPos = nPos[node];
                conv(td,pe*nLeaves);
//                conv(td,pe);

                // The same for flipped version
                td.B    = static_cast<float*>(mxGetData(mxGetCell(cellVarCellFlip,node)));
                td.C    = ubFlip + node*nPixels;
                td.pos  = posFlip + nMaxPos*node;
                td.nPos = nPosFlip[node];
                conv(td,pe*nLeaves);
//                conv(td,pe);
            }
        }
    }
    free(A);
    //free(B);
    mxFree(pos);
    mxFree(posFlip);
    mxFree(nPos);
    mxFree(nPosFlip);
    mxFree(varf);   // necessary if we compute feature variance inside mex code
}






