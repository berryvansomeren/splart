# Splart: Differentiable Painting

Splart is an alternative approach to the ideas of Drarwing.
Instead of the genetic algorithm drarwing uses, 
splart uses differential rendering and backpropagation to generate images.

This approach is inspired by 3D gaussian splatting, 
but in a 2D setting. So 2D gaussian splatting. But it also trades gaussian splats for textures. 
So 2D texture compositing. But that sounds like normal rasterization, while we use differentiable rendering....
Let's call it differentiable painting!

### Installation

Just run `poetry install` in the root directory of this repository for basic functionality.
If you want your gifs to be <span style="color:deepskyblue">*optimized*</span>, 
then you will also need to install [gifsicle](https://formulae.brew.sh/formula/gifsicle)
in addition to the dependencies in pyproject.toml.

---

### Todos:

Add a brush scale multiplier. Loss goes down to 0.11 but by that time, brushes are way too small.
Perhaps x2 is a good start?

Debug sampling using a non-square image! Colors seem not to correspond to sampling locations

Add Redraw
Add automatic target image size selection
Add running batches
Have fun! Create some art!
Use difference image to further guide splat placement


