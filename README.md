# HT_Engine

> **"We don't train models. We define worlds."**

A high-performance Python game engine focused on realistic natural environment simulation.

![HT_Engine Demo](cdd3d7f59f6e2f429f328d4e6c93ecc7.png)

*HT_Engine Demo - 22~26 FPS, orange-yellow sun, dry rock sand terrain (~520K triangles, ~100+ dynamic objects)*

**Rendering Status:**
- Draw Calls: ~100+ (terrain + dynamic objects)
- GPU Instancing: Not enabled yet
- Optimization Used: Simple state batching (grouping by material + mesh)
- Research Direction: CPU-GPU interaction optimization & GPU instancing
- Future Plan: "Community System" self-developed GPU instancing system (coming this summer)

**Model Source:** The terrain model shown in this screenshot is "Dry Rock Sand Terrain" by **josevega**.
- **Author:** josevega
- **Source:** https://www.aigei.com/item/free_dry_rock.html
- **License:** See model license on Aigei.com

The sun effect is procedurally generated using shaders.

**Note:** This demo (`terrain_demo.py`) is specifically customized for this terrain model. If you use other models (e.g., from the MODELS_GUIDE.md), you may need to adjust the code (such as normal flipping, scaling, or material settings) to display them correctly.

**Note on Visuals:** The sun in this demo is procedurally generated using a simplified shader-based approach, created entirely from code without external assets. While AAA games typically use complex sky systems with volumetric lighting, atmospheric scattering simulations, and HDR cubemaps for realistic sun effects, this implementation demonstrates the engine's core rendering capabilities with a lightweight, code-only solution. The focus is on performance and functionality rather than photorealism at this stage.

**Performance Benchmark:**
- **GPU:** NVIDIA GTX 1650 Max-Q (4GB VRAM)
- **FPS:** 22~26 FPS stable (terrain + 100+ dynamic objects)
- **Model:** ~520,000 triangles (Dry Rock Sand Terrain by josevega)
- **Demo:** `terrain_demo.py`
- **Active GPU-Accelerated Features:**
  - GPU Occlusion Culling (Lighting.OcclusionGPU)
  - Screen Space Shadows (Lighting.ScreenSpaceShadows)
  - Enhanced Atmospheric Scattering (Atmosphere.ScatteringEnhanced)
  - Atmospheric Optical Effects (Atmosphere.OpticalGPU)
  - Simple Physics (Physics.SimpleRigidBody)

## Current Status: 0.8 MVP

> **🚨 Urgent Notice from the Author:**
> *"My mom is about to take away my computer, so I'm open-sourcing this now. Enjoy!"*
> 
> — Wang Ruilin, 16, probably grounded soon

**What is this?**

HT_Engine is **0.8 MVP** (Minimum Viable Product). It's functional and you can build with it, but there's still work to do.

**Why 0.8 and not 1.0?**

- The engine is functional but not yet production-ready
- Some features are experimental or incomplete
- Performance optimizations are ongoing
- API may change in future updates

We openly acknowledge our limitations and the "legacy code" (some parts of the codebase need refactoring). The engine was developed with AI assistance for coding, which helped accelerate development but also means some code patterns may not be optimal.

**Our Commitment:**
- We will continue to update, optimize, and improve the engine
- Regular updates will be released as we fix bugs and add features
- Community feedback is welcome and appreciated
- We aim to make this a truly capable engine for indie developers

## Features

- 🌲 **Natural System**: Symbolic AI-based environment simulation
- 🎨 **Advanced Rendering**: OpenGL 4.5 + GPU acceleration
- 🏔️ **Terrain System**: Large-scale terrain rendering support
- 💡 **Lighting System**: Real-time shadows, volumetric light, atmospheric scattering
- 🎮 **Simple API**: Easy-to-use game development interface

## System Requirements

- Python 3.10+
- OpenGL 4.5 compatible graphics card
- Recommended: NVIDIA GTX 1650 or higher

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python play_test_iceland.py
```

## License

**Default:** [GPLv3](LICENSE) (free for open source projects)

**Commercial Use:** Contact us for commercial license
- 📧 Email: 13910593150@139.com
- 💬 WeChat: wxid_pt34d1oc0a0d22

*Commercial license allows closed-source development with mandatory attribution.*

## Contributing & Academic Recognition

### How to Contribute

Issues and Pull Requests are welcome!

### Contributor Benefits

**Immediate Benefits:**
- Contributors retain attribution rights to their code
- Grant the author the right to use the code in all versions
- Recognition in project documentation

**Academic Recognition:**

For contributors who provide significant technical contributions:

| Contribution Level | Recognition |
|-------------------|-------------|
| **Core Algorithm Development** | Co-author (2nd or 3rd author) on HT_Engine research papers |
| **Major Technical Breakthrough** | Co-author on papers related to their contribution |
| **Significant Feature Implementation** | Acknowledged in paper contributions section |
| **Bug Fixes & Documentation** | Listed in project contributors and acknowledgments |

**Criteria for Co-authorship:**
- Direct participation in core algorithm design and implementation
- Providing critical technical insights that significantly improve the engine
- Major breakthrough contributions to rendering, physics, or AI systems
- Sustained contribution over an extended period

**Criteria for Acknowledgments:**
- Code contributions (features, optimizations, bug fixes)
- Documentation improvements
- Community support and mentoring
- Testing and quality assurance
- Technical discussions and suggestions

### Publication Plan

The author is currently exploring the academic potential of HT_Engine's innovative technologies. Should the project itself—or substantial algorithmic contributions from collaborators—yield significant theoretical breakthroughs, particularly in areas such as symbolic AI for environmental simulation, GPU acceleration strategies, or novel rendering techniques, there may be opportunities for collaborative academic publications. Those who provide key algorithmic insights or core technical contributions will naturally be considered for appropriate authorship or recognition in any resulting scholarly work. However, the primary focus remains on building a robust, practical engine for the game development community.

## Contact

- Email: 13910593150@139.com
- WeChat: wxid_pt34d1oc0a0d22

## Acknowledgments

Every contribution to HT_Engine, whether large or small, is deeply appreciated. This project exists because of the collective effort of passionate individuals who believe in democratizing game development technology.

### Core Contributors

*To be updated as the project grows*

### Special Thanks

- All GitHub contributors who submitted code, documentation, or bug reports
- Community members who provided feedback and suggestions
- Early adopters who tested the engine and reported issues
- Technical experts who shared their knowledge and insights
- Everyone who believed in this project and helped it grow

### Technical Advisors

*To be updated as the project grows*

---

**Note**: This acknowledgment list will be continuously updated. If you have contributed to HT_Engine and are not listed here, please contact the author to be added.

---

**Note**: This project is still under active development. APIs may change.

---

## About the Author

![Author Photo](7a1278eb9145f73c708ca103b62da1c7.jpg)

**Wang Ruilin (王瑞霖)** - Creator of HT_Engine

I'm Wang Ruilin, nicknamed **"Hutou" (Tiger Head)**, a 16-year-old high school student from China. I built this game engine in my spare time between classes. I'm passionate about symbolic AI, rendering, and GPU acceleration. I want games that don't just look good—they should feel alive. I also believe in **technology democratization**: stunning visuals shouldn't require expensive hardware. If Unreal Engine can create breathtaking scenes, why can't we achieve something similar on a budget laptop? That's the challenge I'm taking on. Why Python instead of C++? Two answers—high EQ: we're exploring the performance ceiling of interpreted languages. Low EQ: I don't know C++.

---

> **"We don't train models. We define worlds."**
>
> *- Wang Ruilin, Creator of HT_Engine*
