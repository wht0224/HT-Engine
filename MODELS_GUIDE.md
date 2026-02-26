# Model Resources Guide

Due to copyright restrictions, some 3D models are not included in this open-source repository. Below are methods to obtain these models:

## HT_Engine Demo Models

**Source:** Aigei.com (爱给网) and other resources

### Current Demo Model (~520K Triangles)

**Dry Rock Sand Terrain (Currently Used in Demo)**
- **Author:** josevega
- **Source:** https://www.aigei.com/item/free_dry_rock.html
- **Format:** .gltf
- **Description:** High-detail dry rock sand terrain with **~520,000 triangles**. Currently used for the HT_Engine Demo, achieving **49 FPS** on GTX 1650 Max-Q with all GPU-accelerated features.
- **Performance:** 49 FPS stable, 520K triangles, 8+ GPU effects active
- **License:** See model license on Aigei.com

### High-Performance Test Model (720K Triangles)

**Glacier Snow Mountain Model - Iceland Mountain**
- **Author:** Unknown (Aigei.com)
- **Download Link:** https://www.aigei.com/item/bing_chuan_xue_11.html
- **Format:** .gltf / .glb
- **Description:** High-detail terrain model with **720,000 triangles**. HT_Engine has been tested with this model achieving **49 FPS** on GTX 1650 Max-Q with all GPU-accelerated features enabled.
- **Performance:** 49 FPS stable, 720K triangles, 8+ GPU effects active
- **⚠️ Note:** This model may require VIP membership to download. Due to copyright restrictions, it is not included in the open-source repository. Users can download and test themselves to verify HT_Engine's performance capabilities.

### Alternative Test Models

**Snowy Mountain - Terrain (CC-BY 4.0)**
- **Author:** artfromheath
- **Source:** https://www.aigei.com/item/snowy_mountain_131.html
- **Format:** .gltf
- **Description:** Snowy mountain terrain with ~130,000 triangles. Licensed under CC-BY 4.0 (requires attribution).
- **Performance:** 45-49 FPS on GTX 1650 Max-Q
- **License:** CC-BY 4.0

**Mt. Everest - 珠穆朗玛峰**
- **Author:** yyao39
- **Model Generator:** https://elevationapi.com/playground_3dbboxDEM (SRTM_GL1)
- **Texture:** MapBox Satellite, processed by Adobe Lightroom
- **Format:** .gltf
- **Description:** Mt. Everest terrain model
- **License:** See model license on Aigei.com

**Other Free Models**
- See "Free Alternative Resources" section below for CC0 and CC-BY licensed models

**Alternative Search Keywords:**
- "Iceland Mountains" or "冰川雪山"
- "Snow Mountain" or "雪山"
- "Glacier" or "冰川"

### How to Use

1. Visit the link above to download the model file (or use the free alternative resources below)
2. Place the model in the `models/` folder
3. Modify the model path in the code:
   ```python
   model_path = "models/Landscape_Ice.gltf"  # or your downloaded filename
   ```

### About the Sun

The sun effect in this demo (orange-yellow gradient sun) is procedurally generated, not a 3D model. The relevant code is in `terrain_demo.py`, using OpenGL shaders for rendering.

## Included Models

This repository includes the following freely usable models:

- `output/terrain_2x.glb` - Procedurally generated terrain model (ready to use)

## Free Alternative Resources (Recommended)

If you don't want to purchase VIP or are looking for free models, here are recommended free resources:

### Free 3D Model Libraries

**GLB/GLTF Format Specific:**
- **GitHub GLB Models** - Search for "glb free model" or "gltf free model"
- **Khronos Group GLTF Sample Models** - Official sample models
- **Microsoft GLTF Models** - Free models provided by Microsoft

**Comprehensive Free Model Websites:**
- **Poly Haven** (polyhaven.com) - CC0 license, free for commercial use
  - Includes HDR environments, textures, 3D models
  - Search: "mountain", "terrain", "landscape"
- **Sketchfab** (sketchfab.com) - Filter by CC license
  - Search: "free", "CC0", "CC-BY"
- **OpenGameArt** (opengameart.org) - Free resources for games
- **Kenney.nl** - Free game resources, CC0 license
- **Quaternius** (quaternius.com) - Free 3D model packs

**Terrain Specific:**
- **Terrain.Party** - Real-world terrain data
- **Google Earth Studio** - Can export terrain data
- **NASA SRTM Data** - Free elevation data (requires conversion)

## Other Resources

### Textures and Materials

For textures and materials, you can visit:
- Poly Haven (polyhaven.com) - Free CC0 textures
- Texture Haven - Free texture resources
- CC0 Textures - Free commercial textures
- ShareTextures - Free PBR textures

### 3D Model Resources

Free commercial models:
- Sketchfab (filter by CC license)
- TurboSquid (free section)
- CGTrader (free section)
- Free3D (free section)

## Important Notes

1. When using third-party models, please comply with their license agreements
2. For commercial use, please confirm the model license allows commercial use
3. We recommend prioritizing CC0 or CC-BY licensed resources

## Creating Your Own Models

You can also use the following tools to create your own models:
- Blender (free and open-source)
- Gaea (terrain generation)
- World Creator (terrain generation)

After creation, export as .glb or .gltf format for use.

---

For questions, please contact: 13910593150@139.com / wht333@qq.com
