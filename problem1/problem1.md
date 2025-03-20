# Question 1
**Why are the MLP reconstructions so much less detailed than those produced by the SIREN?**

The MLP use ReLU as activation functions, which struggle with high-frequency details and spatial derivatives. The outputs can only approximate continuous signals in a segmented manner, making them incapable of smoothly capturing fine details. Additionally, the second derivatives in ReLU are zero everywhere, which makes them ineffective at modeling physical signals that rely on differential equations.

SIRENs, on the other hand, leverage sinusoidal activation functions, which inherently capture high-frequency features and allow the model to smoothly interpolate signals. Their second-order derivatives remain well-structured, enabling better image reconstruction.

# Question 2
**Why does the image Laplacian produced by the MLP look strange?**

Because MLPs struggle to represent smooth, high-frequency variations in an image. The Laplacian (second derivative) measures curvature, and since ReLU networks lack smooth second derivatives, their Laplacians are noisy or unnatural.