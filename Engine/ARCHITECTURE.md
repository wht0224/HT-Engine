# HT_Engine Architecture Overview

## Core Philosophy

HT_Engine is built around a simple idea: **symbolic AI for environment simulation**. Instead of hard-coding every environmental interaction, we use a rule-based system that mimics how nature actually works.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Game Code                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    HT_Engine Core                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Engine    │  │   Scene     │  │   Resource Manager  │  │
│  │   (Main)    │  │   Manager   │  │   (Assets/VRAM)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Rendering Pipeline                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Renderer  │  │   Shaders   │  │   Post-Processing   │  │
│  │   (OpenGL)  │  │   (GLSL)    │  │   (Effects)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              The Natural System (Symbolic AI)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Rules     │  │   Facts     │  │   Inference Engine  │  │
│  │   (Logic)   │  │   (State)   │  │   (Reasoning)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              GPU-Accelerated Rule Execution                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Shader    │  │   GPU       │  │   Compute           │  │
│  │   Fragments │  │   Context   │  │   (Transform)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Engine Core (`Engine/`)
The heart of the system. Manages initialization, main loop, and coordinates all subsystems.

### 2. Natural System (`Engine/Natural/`)
Our unique approach to environment simulation:
- **Rules**: Define how nature behaves (lighting, atmosphere, physics)
- **Facts**: Current state of the environment
- **Inference Engine**: Applies rules to facts to generate behavior
- **GPU Execution**: Rules compile to shaders for real-time performance

### 3. Rendering Pipeline (`Engine/Renderer/`)
- **Deferred/Forward/Hybrid**: Multiple rendering paths for different needs
- **Effects**: Bloom, SSAO, Volumetric Light, Motion Blur, etc.
- **Optimization**: Frustum culling, LOD, GPU capability evaluation

### 4. Scene Management (`Engine/Scene/`)
- Hierarchical scene graph
- Camera control
- Lighting
- Object management

### 5. Platform Abstraction (`Engine/Platform/`)
- Cross-platform windowing
- Input handling
- OpenGL context management

## How It Works

1. **You create a scene** using the SceneManager
2. **You add objects** (meshes, lights, cameras)
3. **The Natural System** automatically handles:
   - Lighting interactions
   - Atmospheric effects
   - Physics (simplified)
   - Environmental responses
4. **Rules compile to GPU shaders** for real-time performance
5. **The Renderer** draws everything efficiently

## Why Python?

We know, we know. "Python for a game engine? Are you crazy?"

Turns out, with proper architecture:
- Python handles high-level logic beautifully
- Numba/Cython handle math-heavy operations
- OpenGL handles rendering
- Result: 49 FPS with 520K triangles on a laptop GPU

## Performance Secrets

1. **GPU Rule Execution**: Environmental logic runs on GPU, not CPU
2. **Frustum Culling**: Only render what's visible
3. **LOD System**: Simplify distant objects
4. **VRAM Management**: Intelligent texture/model streaming
5. **Shader Optimization**: Generated shaders are optimized per-rule

## Getting Started

See `QUICKSTART.md` for setup instructions.

See `terrain_demo.py` for a working example.

---

