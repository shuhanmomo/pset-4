# 6.830-pset4 Problem 2: Sphere Tracing and Volumetric Rendering

## Problem 2.1: Sphere Tracing 

Implicit surfaces are a powerful representation for 3D shapes where the surface is defined as the level set of a function. In this assignment, you will implement **sphere tracing**, an efficient ray marching technique for rendering implicit surfaces defined by signed distance functions (SDFs).

Unlike volumetric rendering that samples points along the entire ray, sphere tracing leverages the SDF property to take larger steps where it's safe to do so, making it much more efficient for rendering implicit surfaces.

## Problem Description

You will implement a sphere tracing renderer for a simple scene containing two spheres. The scene is defined using an implicit model that returns the SDF value and color for any 3D point. Your task is to:

1. **Implement camera ray generation** (reuse from the previous problem)
2. **Implement sphere tracing** to find intersections between rays and the implicit surface
3. **Generate a rendered image** by combining the above components

## Part A: Camera Rays

This function should:
- Take camera-to-world transform and intrinsic parameters
- Generate ray origins and directions for each pixel in the image
- You need to convert pixel coordinates to camera space using the provided intrinsic parameters `[fx, fy, cx, cy]`
- The resulting rays start at the camera position (origin) and go through each pixel. (remember to shift the x,y cords in camera plane by +0.5)
- Use the camera-to-world transform `c2w` to transform rays from camera space to world space
- Return ray origins and directions with shape `[H, W, 3]`

Hint: The camera rays in camera space are defined as follows; be careful to normalize them to unit length later: :
```
X_cam = (x - cx) / fx
Y_cam = (y - cy) / fy
Z_cam = 1
```

Remember to normalize the ray_dir.


## Part B: Sphere Tracing

Implement the `sphere_tracing` function, which should:
1. Initialize each ray with a starting **t** value 
2. March along each ray by querying the SDF at the current point
3. For each step, advance the ray by a distance equal to the SDF value (this is safe as the SDF tells us the distance to the closest surface)
4. Stop marching when either:
   - The ray hits the surface (SDF < epsilon)
   - The maximum number of iterations is reached
   - The ray goes too far (beyond **t_far**)
5. For each ray that hits a surface, record the color at the hit point

## Hints

### Sphere Tracing Algorithm

1. **Key Insight**: The SDF value at any point tells you how far you can safely move along the ray without missing any surface.

2. **Implementation Steps**:
   - Initialize a tensor `t` for each ray's current position
   - Track which rays have hit a surface with a boolean mask
   - For each iteration:
     - Sample points using the current **t** values: `points = origins + t * directions`
     - Query the SDF model to get distance values and colors
     - Determine which rays hit a surface (SDF < epsilon)
     - Record colors for rays that hit
     - For rays that haven't hit yet, advance t by the SDF value (optionally using a relaxation factor like 0.5)

3. **Batched Processing**: Process all rays in parallel for efficiency.

4. **Error Checking**: Handle cases where rays might never hit any surface.

### Testing
After implementing all the required functions, you can test your implementation by running:
```
python problem2/problem_2_sphere_tracing.py
```

### Expected Output

When correctly implemented, your renderer should show two spheres (blue and green) from two different camera angles. The spheres should appear properly shaded based on their colors. I’ve put the expected renders for the debugging intrinsics in `expected_renders_for_debug/sphere_tracing.png`.

## Mathematical Background

Sphere tracing works because of the properties of signed distance functions:
- An SDF returns the minimum distance to any surface.
- If we move a distance d along a ray where d is the SDF value at the current point, we're guaranteed not to miss any surface.

This allows for efficient rendering of implicit surfaces without the need for dense sampling along the entire ray.


## Problem 2.2: Differentiable Volume Rendering

### Overview

In this problem, you will implement a differentiable volume renderer for 3D implicit functions. The renderer will allow you to visualize a 3D scene defined by an implicit function from different camera views. This technique forms the foundation of Neural Radiance Fields (NeRF) and other neural rendering approaches.

### Task Description
You are provided with an implementation of a 3D implicit function that represents a scene containing multiple geometric shapes (spheres and cubes) with different colors. Your task is to implement the following components of a volumetric renderer:

1. **Camera Ray Generation (Part A)** - Implement the `camera_param_to_rays` function to generate rays from a camera.
2. **Ray Marching (Part B)** - Implement the `sample_points_on_rays` function to sample points along each ray.
3. **Volumetric Rendering (Part C)** - Implement the `volume_rendering` function to compute the final color for each ray.

The file `problem2/problem_2_volumetric_rendering.py` contains a template with the necessary function signatures and documentation comments. You need to complete the TODO sections in these functions.

### Implementation Details and Hints

#### Part A: Camera Ray Generation

Reuse your implementation of `camera_param_to_rays` from the sphere tracing problem. 


#### Part B: Ray Marching
- Generate `num_samples` points along each ray between `t_near` and `t_far`.
- The sample points are defined as: `point = origin + t * direction` for each value of `t`.
- Return an array of sampled points with shape `[H, W, num_samples, 3]`.


#### Part C: Volumetric Rendering
- Implement the classical volume rendering equation: 
  C = ∑ T_i * α_i * c_i, where:
  - T_i = ∏_{j=1}^{i-1} (1 - α_j) is the accumulated transmittance
  - α_i = 1 - exp(-σ_i * δ_i) is the alpha value at sample i
  - σ_i is the density at sample i
  - δ_i is the distance between adjacent samples
  - c_i is the color at sample i
For more details elaboration on this equation, please refer to NeRF's original paper: https://arxiv.org/abs/2003.08934 (Eqs. 1, 2, 3, and 5 would be helpful).


Hint: Iterate through samples front-to-back (from **t_near** to **t_far**), accumulating color and updating transmittance.

### Testing
After implementing all the required functions, you can test your implementation by running:
```
python problem2/problem_2_volumetric_rendering.py
```

The script will render the scene from two different camera viewpoints and display the results. The correct rendering should show multiple colored shapes (blue and green spheres, red and yellow cubes) in 3D space. The rendering result of the debugging intrinsics should look like `problem2/expected_renders_for_debug/volume_rendering.png`.

### Please answer the following questions:
Nowadays, the majority of differentiable rendering uses volumetric rendering rather than sphere tracing. Can you think of some reasons why? (There might be multiple reasons, try to think at least one)


### Extra Challenge
- Try changing the number of samples along the rays and observe the effect on rendering quality.
- Sampling is a very important part not only in volume rendering but also in the whole field of rendering. Try implementing stratified sampling instead of uniform sampling.

Happy rendering!

## Submission guidelines

1. For the code, submit the two Python files. Also, include the edited code in the report PDF file. (either paste formatted code or screenshot)

2. Please submit your rendered images for both problems in the report pdf file. Please make sure that you used the intrinsics corresponding to the `submit` mode.

3. For the questions, write in English and submit the report PDF file.

It's better to name this pdf as problem_2.pdf. If you want to combine it with the pdf report from problem_3, it's fine as well. 
