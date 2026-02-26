# Quick Start

## Running the Terrain Demo

### 1. Model Already Included

The demo includes a **Dry Rock Sand Terrain** model (~520K triangles) by josevega.

**Model Info:**
- **Author:** josevega
- **Source:** https://www.aigei.com/item/free_dry_rock.html
- **License:** See model license on Aigei.com

The model is already extracted in `models/干燥岩砂地形/`.

### 2. Alternative Models (Optional)

You can also test with other models:
- **Snowy Mountain** (~130K triangles, CC-BY 4.0) by artfromheath
- **Iceland Mountain** (~720K triangles) - requires VIP download from Aigei.com

See `MODELS_GUIDE.md` for details.

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. Run the Demo

```bash
python terrain_demo.py
```

## No Model Required

The demo already includes a working terrain model. Just run the command above!

## Troubleshooting

**Issue: Model fails to load**
- Check if model files exist in `models/干燥岩砂地形/`
- Verify file permissions

**Issue: Missing dependencies**
- Run `pip install -r requirements.txt`
- Ensure Python version >= 3.10

**Issue: OpenGL errors**
- Ensure GPU supports OpenGL 4.5
- Update graphics drivers

## More Help

- See [MODELS_GUIDE.md](MODELS_GUIDE.md) for more model resources
- See [README.md](README.md) for full documentation
- Contact: 13910593150@139.com / wht333@qq.com
