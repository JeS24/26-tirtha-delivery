# Tirtha – Gaussian Splatting Updates

**NSIP 2026 Summer Internship**

**Intern:** M Meghna Monalee  
**Mentor:** Jyotirmaya Shivottam  
**Supervisor:** Dr. Subhankar Mishra  
**Organization:** National Institute of Science Education and Research (NISER), Bhubaneswar

---

## Project Overview

This project focused on investigating efficient post-processing techniques for **3D Gaussian Splatting (3DGS)** scenes as part of the **Tirtha** project. The work involved evaluating compression, simplification, streaming, and collision-generation pipelines while preserving rendering quality and enabling efficient visualization.

The experiments were performed using **NanoGS** and **PlayCanvas Splat-Transform** across multiple large-scale Gaussian Splatting scenes.

---

# Main Outcomes

## 1. NanoGS Simplification

- Evaluated Gaussian simplification at multiple reduction ratios (75%, 50%, 25%, and 10%).
- Compared simplification quality and runtime.
- Analyzed memory usage and resulting Gaussian counts.
- Benchmarked different neighborhood sizes (k = 32 and k = 64).

---

## 2. SOG Compression

- Converted Gaussian Splatting scenes into the **SOG (Self-Organizing Gaussians)** format.
- Measured:
  - Compression ratio
  - Runtime
  - Memory usage
  - Output size
- Achieved approximately **14×–17× reduction** in storage compared to the original PLY files.

---

## 3. Filtering Experiments

Investigated built-in filtering operations provided by Splat-Transform:

- Floater removal
- Cluster filtering
- Combined filtering

Compared filtered outputs with the original scenes to study their effect on scene quality and storage.

---

## 4. Level-of-Detail (LoD) Generation

Generated streamed LoD representations for multiple scenes.

Evaluated:

- Chunk generation
- LoD hierarchy
- Gaussian counts
- Storage overhead
- Streaming metadata

---

## 5. Format Conversion

Generated multiple output formats from Gaussian Splatting scenes including:

- SOG
- GLB
- HTML Viewer
- Voxel metadata

Compared runtime and output sizes for each format.

---

## 6. Voxelization and Collision Mesh Generation

Generated voxel representations and collision meshes for Gaussian Splatting scenes.

Investigated:

- Octree statistics
- Voxel resolution
- Collision mesh generation
- Viewer compatibility

---

## 7. Performance Evaluation

Collected detailed benchmarking information including:

- Runtime
- Peak memory usage
- Compression ratios
- Gaussian counts
- Storage requirements

across several real-world scenes.

---

# Tools Used

- NanoGS
- PlayCanvas Splat-Transform
- SuperSplat Viewer
- Python
- Linux
- Git
- LaTeX

---

# Repository Contents

```
reports/         Weekly and final reports
presentation/    Final presentation
scripts/         Experiment scripts
docs/            Experiment documentation
results/         Generated outputs and benchmark tables
images/          Figures and screenshots
```

---

# Key Takeaways

- Successfully evaluated multiple post-processing pipelines for Gaussian Splatting.
- Demonstrated significant storage reduction using SOG compression.
- Generated multi-resolution LoD representations for efficient streaming.
- Explored voxelization and collision generation for interactive applications.
- Benchmarked performance across multiple large-scale 3DGS datasets.

---

**National Summer Internship Programme (NSIP) 2026**  
**National Institute of Science Education and Research (NISER), Bhubaneswar**

