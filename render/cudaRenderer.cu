#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

// constant variable tile size
#define TILE_SIZE 4
// nvidia-smi | grep 'render' | awk '{print $5}' | xargs -n1 kill -9

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////
#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

// This stores the global constants
struct GlobalConstants
{

    SceneName sceneName;

    int numberOfCircles;

    float *position;
    float *velocity;
    float *color;
    float *radius;

    int imageWidth;
    int imageHeight;
    float *imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int cuConstNoiseYPermutationTable[256];
__constant__ int cuConstNoiseXPermutationTable[256];
__constant__ float cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];

// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake()
{

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a)
{

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks()
{
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;
    float *radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS)
    { // firework center; no update
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i + 1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j + 1] += velocity[index3j + 1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j + 1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist)
    { // restore to starting position
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        // Travel scaled unit length
        velocity[index3j] = cosA / 5.0;
        velocity[index3j + 1] = sinA / 5.0;
        velocity[index3j + 2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff)
    {
        radius[index] = 0.02f;
    }
    else
    {
        radius[index] += 0.01f;
    }
}

// kernelAdvanceBouncingBalls
//
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls()
{
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    if (oldVelocity == 0.f && oldPosition == 0.f)
    { // stop-condition
        return;
    }

    if (position[index3 + 1] < 0 && oldVelocity < 0.f)
    { // bounce ball
        velocity[index3 + 1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3 + 1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3 + 1] += velocity[index3 + 1] * dt;

    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon && oldPosition < 0.0f && fabsf(position[index3 + 1] - oldPosition) < epsilon)
    { // stop ball
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake()
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float *positionPtr = &cuConstRendererParams.position[index3];
    float *velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3 *)positionPtr);
    float3 velocity = *((float3 *)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ((position.y + radius < 0.f) ||
        (position.x + radius) < -0.f ||
        (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3 *)positionPtr) = position;
    *((float3 *)velocityPtr) = velocity;
}

// function for shared memory indexing
__device__ __inline__ int getIndex(int x, int y, int thread_id)
{
    return thread_id + (x * blockDim.x) + (y * blockDim.x * TILE_SIZE);
}

__device__ __inline__ int getIndex(int x, int y)
{
    return x + (y * TILE_SIZE);
}
__device__ void printSharedMemory(float4 *sharedMem, int index)
{
    for (int i = 0; i < blockDim.x; i++)
    {
        printf("Value at index %d of shared memory: %f\n", i, sharedMem[i].x);
    }
}

__device__ __inline__ int
circleContribute(float2 pixelCenter, float3 p, int circleIndex)
{

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return 0;
    else
        return 1;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ void
shadePixel(float2 pixelCenter, float3 p, float4 *imagePtr, int circleIndex, int pixelX, int pixelY, float4 *sMem)
{

    // flag for circle contribution if flag == 1, circle contributes to pixel
    if (!circleContribute(pixelCenter, p, circleIndex))
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME)
    {
        float diffX = p.x - pixelCenter.x;
        float diffY = p.y - pixelCenter.y;
        float pixelDist = diffX * diffX + diffY * diffY;

        float rad = cuConstRendererParams.radius[circleIndex];
        float maxDist = rad * rad;

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
    }
    else
    {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3 *)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    // int index = getIndex(pixelX, pixelY, threadIdx.x);
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     printSharedMemory(sMem, 10);
    // }

    // sMem[index].x = rgb.x;
    // sMem[index].y = rgb.y;
    // sMem[index].z = rgb.z;
    // sMem[index].w = 1;

    // if (circleIndex == 0)
    // {
    //     if (index == 345291)
    //     {
    //         printf("red %f redShared\n", rgb.x);
    //         printf("red %f redShared %f\n", rgb.x, sMem[index].x);
    //     }
    //     float4 existingColor = *imagePtr;
    //     float4 newColor;
    //     newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    //     newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    //     newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    //     newColor.w = alpha + existingColor.w;

    //     // Global memory write
    //     *imagePtr = newColor;
    // }

    // END SHOULD-BE-ATOMIC REGION
}

__device__ __inline__ void
shadeScreen(int screenMinX, int screenMaxX, int screenMinY, int screenMaxY,
            int tileMinX, int tileMinY,
            float invWidth, float invHeight, int imageWidth,
            int imageHeight, float3 p, int circleIndex, short *sharedMemory)
{
    // For all pixels in the bounding box
    for (int pixelY = screenMinY; pixelY < screenMaxY; pixelY++)
    {
        float4 *imgPtr = (float4 *)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX = screenMinX; pixelX < screenMaxX; pixelX++)
        {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));

            // x,y in the tile:
            int tileX = pixelX - tileMinX;
            int tileY = pixelY - tileMinY;
            int index = getIndex(tileX, tileY, threadIdx.x);
            // printf("t:(%d, %d) p:(%d, %d) index: %d\n", tileX, tileY, pixelX, pixelY, index);

            if (circleContribute(pixelCenterNorm, p, circleIndex))
            {
                if (pixelX == 415 && pixelY == 562)
                    printf("Pixel (%d, %d) contributes to circle %d index %d\n", pixelX, pixelY, circleIndex, index);
                sharedMemory[index] = 1;
            }
            else
            {
                sharedMemory[index] = 0;
            }

            // shadePixel(pixelCenterNorm, p, imgPtr, circleIndex, pixelX, pixelY, sMem);
            imgPtr++;
        }
    }
}

__device__ void
performScan(short *sharedMemory, int flag)
{
    short N = 256 * TILE_SIZE * TILE_SIZE;
    if (threadIdx.x == 0)
    {
        sharedMemory[N] = sharedMemory[N - 1];
    }

    // up-sweep
    for (int twod = 1; twod < N; twod *= 2)
    {
        __syncthreads();
        int twod1 = twod * 2;
        for (int index = threadIdx.x * twod1; index < N; index += blockDim.x * twod1)
        {

            if (flag == 1)
                printf("m[%d] = %d, m[%d] = %d\n", index + twod - 1, sharedMemory[index + twod - 1], index + twod1 - 1, sharedMemory[index + twod1 - 1]);
            sharedMemory[index + twod1 - 1] += sharedMemory[index + twod - 1];
        }
    }
    if (threadIdx.x == 0)
    {
        sharedMemory[N - 1] = 0;
    }

    // down-sweep
    for (int twod = N / 2; twod >= 1; twod /= 2)
    {
        int twod1 = twod * 2;
        __syncthreads();

        for (int index = threadIdx.x * twod1; index < N; index += blockDim.x * twod1)
        {
            int tmp = sharedMemory[index + twod - 1];
            sharedMemory[index + twod - 1] = sharedMemory[index + twod1 - 1];
            sharedMemory[index + twod1 - 1] += tmp;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        sharedMemory[N] += sharedMemory[N - 1];
}

__device__ void
markCircle(int index, short *sharedMemory, short tileMinX, short tileMinY, short imageWidth, short imageHeight)
{
    // Read position and radius
    int index3 = 3 * index;
    float3 p = *(float3 *)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[index];

    printf("index: %d p = (%f, %f, %f) rad = %f\n", index, p.x, p.y, p.z, rad);

    // Compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    short tileMaxX = min(tileMinX + TILE_SIZE, imageWidth);
    short tileMaxY = min(tileMinY + TILE_SIZE, imageHeight);

    short screenMinX = max(tileMinX, min(minX, tileMaxX));
    short screenMaxX = max(tileMinX, min(maxX, tileMaxX));
    short screenMinY = max(tileMinY, min(minY, tileMaxY));
    short screenMaxY = max(tileMinY, min(maxY, tileMaxY));

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    // 415, 562
    // Tile - (412, 560) - (416, 564)
    // if (threadIdx.x == 0)
    // printf("Tile - (%d, %d) - (%d, %d)\n", tileMinX, tileMinY, tileMaxX, tileMaxY);
    // printf("Screen - (%d, %d) - (%d, %d)\n", screenMinX, screenMinY, screenMaxX, screenMaxY);

    shadeScreen(tileMinX, tileMaxX, tileMinY, tileMaxY,
                tileMinX, tileMinY,
                invWidth, invHeight, imageWidth, imageHeight,
                p, index, sharedMemory);
}

__device__ __inline__ void
markZero(short *sharedMemory)
{
    // printf("MarkingZero | Thread: %d\n", threadIdx.x);
    for (int tileX = 0; tileX < TILE_SIZE; tileX++)
    {
        for (int tileY = 0; tileY < TILE_SIZE; tileY++)
        {
            int index = getIndex(tileX, tileY, threadIdx.x);
            sharedMemory[index] = 0;
        }
    }
}

__device__ __inline__ int3
circleIndexToPosition(int index)
{
    int3 p;
    // p.x -> pixelX, p.y -> pixelY, p.z -> circleIndex
    p.z = index % blockDim.x;
    p.y = index / (blockDim.x * TILE_SIZE);
    p.x = (index / blockDim.x) % TILE_SIZE;
    return p;
}

__device__ float4
getRGB(int circleIndex)
{
    float4 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME)
    {

        // const float kCircleMaxAlpha = .5f;
        // const float falloffScale = 4.f;

        // float normPixelDist = sqrt(pixelDist) / rad;
        // rgb = lookupColor(normPixelDist);

        // float maxAlpha = .6f + .4f * (1.f - p.z);
        // maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        // alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
    }
    else
    {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        float3 tempRGB = *(float3 *)&(cuConstRendererParams.color[index3]);
        alpha = .5f;

        rgb.x = alpha * tempRGB.x;
        rgb.y = alpha * tempRGB.y;
        rgb.z = alpha * tempRGB.z;
        rgb.w = alpha;

        return rgb;
    }
}

// __device__ void
// scatter(short *sharedMemory, short *pixels)
// {
//     const int N = 256 * TILE_SIZE * TILE_SIZE;
//     __shared__ int shared_index;
//     if (threadIdx.x == 0)
//         shared_index = 1;
//     __syncthreads();

//     int local_index = atomicAdd(&shared_index, 1);

//     // __shared__ float4 sharedImage[256 * TILE_SIZE * TILE_SIZE + 1];
//     const short coloredPixels = sharedMemory[N];

//     // shared memory for pixels
//     __shared__ int numPixels[TILE_SIZE * TILE_SIZE];
//     if (coloredPixels > 1000)
//     {
//         printf("coloredPixels: %d\n", coloredPixels);
//     }

//     while (local_index <= N)
//     {
//         int3 p;
//         p = circleIndexToPosition(local_index);

//         if (p.z == 0)
//         {
//             int index = getIndex(p.x, p.y);
//             numPixels[index] = sharedMemory[local_index] - sharedMemory[local_index - blockDim.x];
//         }

//         if (sharedMemory[local_index] - sharedMemory[local_index - 1])
//         {
//             int circleIndex = blockDim.x * blockIdx.x + p.z - 1;
//             // printf("circleIndex: %d\n", circleIndex);
//             // float3 p = make_int2(tileMinX + p.x, tileMinY + p.y);
//             pixels[sharedMemory[local_index - 1]] = getRGB(circleIndex);
//             printf("rgb: %f, %f, %f, l: %d\n", pixels[sharedMemory[local_index - 1]].x, pixels[sharedMemory[local_index - 1]].y, pixels[sharedMemory[local_index - 1]].z, local_index);
//             // printf("p(%d, %d, %d) index: %d local: %d\n", p.x, p.y, p.z, index, local_index);
//         }
//         local_index = atomicAdd(&shared_index, 1);
//     }
// }



__device__ void
reduction(short *pixels, short *sharedMemory, short tileMinX, short tileMinY)
{
    // For all pixels in the bounding box
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (TILE_SIZE * TILE_SIZE))
        return;
    
    int startIndex = index * blockDim.x;
    int endIndex = startIndex + blockDim.x;

    int numCircles = sharedMemory[endIndex] - sharedMemory[startIndex];

    int offset = 4* (tileMinY * cuConstRendererParams.imageWidth + tileMinX);

    float4 rgb = make_float4(1.f,1.f,1.f,1.f);
    for (int i=sharedMemory[startIndex]; i<sharedMemory[endIndex]; i++){
        int circleIndex = blockDim.x * blockIdx.x + i%blockDim.x - 1;

        printf("startIndex: %d, i: %d, circleIndex: %d\n", startIndex, i, circleIndex);
        // rgb.w += pixels[startIndex + i].w;
        // rgb.x = pixels[startIndex + i].x + (1 - pixels[startIndex + i].w) * rgb.x;
        // rgb.y = pixels[startIndex + i].y + (1 - pixels[startIndex + i].w) * rgb.y;
        // rgb.z = pixels[startIndex + i].z + (1 - pixels[startIndex + i].w) * rgb.z;
    }
    // *(float4 *)(&cuConstRendererParams.imageData[offset]) = rgb;
    printf("offset: %d, p(%f,%f,%f,%f)\n", offset, rgb.x, rgb.y, rgb.z, rgb.w);
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles()
{
    __shared__ short sharedMemory[TILE_SIZE * TILE_SIZE * 256 + 1]; // Declare shared memory in global kernel

    int drawCircle = 1;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const short imageWidth = cuConstRendererParams.imageWidth;
    const short imageHeight = cuConstRendererParams.imageHeight;

    // for tile in imagewidth and tile in imageheight
    for (short tileMinX = 0; tileMinX < imageWidth; tileMinX += TILE_SIZE)
    {
        for (short tileMinY = 0; tileMinY < imageHeight; tileMinY += TILE_SIZE)
        {
            if (tileMinX == 412 && tileMinY == 560)
            {
            }
            else
            {
                continue;
            }

            short tileMaxX = min(tileMinX + TILE_SIZE, imageWidth);
            short tileMaxY = min(tileMinY + TILE_SIZE, imageHeight);

            if (index < cuConstRendererParams.numberOfCircles)
                markCircle(index, sharedMemory, tileMinX, tileMinY, imageWidth, imageHeight);
            else
                markZero(sharedMemory);

            __syncthreads();
            if (tileMinX == 412 && tileMinY == 560 && threadIdx.x == 0)
            {
                printf("Tile - (%d, %d) - (%d, %d)\n", tileMinX, tileMinY, tileMaxX, tileMaxY);
                // printf("Screen - (%d, %d) - (%d, %d)\n", screenMinX, screenMinY, screenMaxX, screenMaxY);
                printf("Pixel (415, 562) contributes to circle %d\n", index);
                for (int i = 0; i < 4; i++)
                {
                    int index = getIndex(3, 2, i);
                    printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
                }
                int index = getIndex(3, 2, 0);
                printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
                index = getIndex(3, 3, 255);
                printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
                index = getIndex(3, 3, 255) + 1;
                printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
            }
            __syncthreads();

            // // perform a scan on the shared memory for each pixel in the tile
            int flag = 0;
            // if (tileMinX == 412 && tileMinY == 560)
            //     flag = 1;
            performScan(sharedMemory, flag);
            __syncthreads();

            if (tileMinX == 412 && tileMinY == 560 && threadIdx.x == 0)
            {
                printf("Tile - (%d, %d) - (%d, %d)\n", tileMinX, tileMinY, tileMaxX, tileMaxY);
                // printf("Screen - (%d, %d) - (%d, %d)\n", screenMinX, screenMinY, screenMaxX, screenMaxY);
                printf("Pixel (415, 562) contributes to circle %d\n", index);
                for (int i = 0; i < 4; i++)
                {
                    int index = getIndex(3, 2, i);
                    printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
                }
                int index = getIndex(3, 2, 0);
                printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
                index = getIndex(3, 3, 255);
                printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
                index = getIndex(3, 3, 255) + 1;
                printf("sharedMemory[%d]: %d\n", index, sharedMemory[index]);
            }

            __shared__ short pixels[1000];
            // scatter(sharedMemory, pixels);
            // __syncthreads();
            
            reduction(pixels, sharedMemory, tileMinX, tileMinY);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer()
{
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer()
{

    if (image)
    {
        delete image;
    }

    if (position)
    {
        delete[] position;
        delete[] velocity;
        delete[] color;
        delete[] radius;
    }

    if (cudaDevicePosition)
    {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image *
CudaRenderer::getImage()
{

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void CudaRenderer::loadScene(SceneName scene)
{
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void CudaRenderer::setup()
{

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("NVIDIA GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int *permX;
    int *permY;
    float *value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void CudaRenderer::allocOutputImage(int width, int height)
{

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void CudaRenderer::clearImage()
{

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME)
    {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    }
    else
    {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void CudaRenderer::advanceAnimation()
{
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES)
    {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    }
    else if (sceneName == BOUNCING_BALLS)
    {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    }
    else if (sceneName == HYPNOSIS)
    {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    }
    else if (sceneName == FIREWORKS)
    {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void CudaRenderer::render()
{
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaCheckError(cudaDeviceSynchronize());
}
