"""
挪威峡湾场景编辑器 - 全新版本
Norway Fjord Scene Editor - Brand New Version
"""

import numpy as np
import json
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from Engine.Natural.Terrain import TerrainEnhancer
from tools.terrain_texture_system import TerrainTextureSystem


class NorwayFjordEditor:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        
        self.terrain_vbo = None
        self.terrain_ibo = None
        self.terrain_vertex_count = 0
        self.terrain_bounds = None
        self.elevation_data = None
        
        self.camera_pos = np.array([5000.0, 3000.0, 8000.0])
        self.camera_target = np.array([5000.0, 500.0, 5000.0])
        self.camera_distance = 5000.0
        self.camera_yaw = -30.0
        self.camera_pitch = -25.0
        
        self.move_speed = 500.0
        self.rotate_speed = 0.3
        
        self.objects = []
        self.selected_type = 0
        self.object_types = ["rock", "church", "turf_house", "waterfall", "geyser", "boat"]
        
        self.scene_path = r"e:\新建文件夹 (3)\output\iceland_scene.json"
        self.models_path = r"e:\新建文件夹 (3)\output\models"
        
        self.wireframe = False
        self.show_help = True
        
        self.model_cache = {}
        
    def init(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Iceland Scene Editor")
        
        print("=" * 60)
        print("Iceland Scene Editor")
        print("=" * 60)
        print(f"OpenGL: {glGetString(GL_VERSION).decode()}")
        print(f"GPU: {glGetString(GL_RENDERER).decode()}")
        print("=" * 60)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        glClearColor(0.5, 0.7, 0.9, 1.0)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 10, 100000)
        glMatrixMode(GL_MODELVIEW)
        
        self._load_terrain()
        self._load_models()
        self._load_scene()
        
        self._update_camera()
        
        self._print_controls()
        
    def _compute_slope(self, elevation):
        """计算坡度"""
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        slope = slope / (slope.max() + 0.001)
        return slope
    
    def _compute_hillshade(self, elevation, azimuth=315, altitude=45):
        """计算山影"""
        dy, dx = np.gradient(elevation)
        
        az_rad = np.radians(azimuth)
        alt_rad = np.radians(altitude)
        
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        
        hillshade = (np.cos(alt_rad) * np.cos(slope) + 
                     np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
        
        hillshade = np.clip(hillshade, 0, 1)
        return hillshade
        
    def _load_terrain(self):
        print("\n[Loading Iceland Terrain]")
        
        iceland_dem_path = r"e:\新建文件夹 (3)\DEM数据\Iceland\iceland_south_dem.npy"
        
        if os.path.exists(iceland_dem_path):
            print("  Loading Iceland DEM...")
            full_elevation = np.load(iceland_dem_path)
            
            step = 4
            self.elevation_data = full_elevation[::step, ::step]
            
            height, width = self.elevation_data.shape
            print(f"  DEM: {width}x{height} (downsampled from {full_elevation.shape[1]}x{full_elevation.shape[0]})")
            print(f"  Elevation: {self.elevation_data.min():.1f}m ~ {self.elevation_data.max():.1f}m")
            
            slope = self._compute_slope(self.elevation_data)
            hillshade = self._compute_hillshade(self.elevation_data)
            results = {'slope': slope, 'hillshade': hillshade}
        else:
            print("  Iceland DEM not found, generating...")
            from tools.iceland_dem_fetcher import generate_procedural_iceland_dem
            self.elevation_data = generate_procedural_iceland_dem()
            slope = self._compute_slope(self.elevation_data)
            hillshade = self._compute_hillshade(self.elevation_data)
            results = {'slope': slope, 'hillshade': hillshade}
            height, width = self.elevation_data.shape
        
        print("\n[Generating Iceland Textures]")
        texture_system = TerrainTextureSystem(256)
        texture_system.generate_procedural_textures()
        
        slope = results.get('slope')
        aspect = results.get('aspect')
        hillshade = results.get('hillshade')
        
        terrain_colors = texture_system.compute_terrain_colors(
            self.elevation_data, slope, aspect, hillshade
        )
        
        scale = 30.0
        vertices = []
        colors = []
        
        for z in range(height):
            for x in range(width):
                vx = x * scale
                vy = self.elevation_data[z, x]
                vz = z * scale
                vertices.append([vx, vy, vz])
                
                color = terrain_colors[z, x]
                colors.append([color[0], color[1], color[2]])
        
        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        
        indices = []
        for z in range(height - 1):
            for x in range(width - 1):
                i0 = z * width + x
                i1 = z * width + (x + 1)
                i2 = (z + 1) * width + x
                i3 = (z + 1) * width + (x + 1)
                indices.extend([i0, i2, i1, i1, i2, i3])
        indices = np.array(indices, dtype=np.uint32)
        
        self.terrain_bounds = {
            'min': vertices.min(axis=0),
            'max': vertices.max(axis=0),
            'center': vertices.mean(axis=0)
        }
        
        vdata = np.hstack([vertices, colors]).astype(np.float32)
        
        self.terrain_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.terrain_vbo)
        glBufferData(GL_ARRAY_BUFFER, vdata.nbytes, vdata, GL_STATIC_DRAW)
        
        self.terrain_ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.terrain_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        self.terrain_vertex_count = len(indices)
        
        print(f"  Vertices: {len(vertices)}")
        print(f"  Triangles: {len(indices) // 3}")
        print(f"  Bounds: X[{self.terrain_bounds['min'][0]:.0f}~{self.terrain_bounds['max'][0]:.0f}] "
              f"Y[{self.terrain_bounds['min'][1]:.0f}~{self.terrain_bounds['max'][1]:.0f}] "
              f"Z[{self.terrain_bounds['min'][2]:.0f}~{self.terrain_bounds['max'][2]:.0f}]")
        
    def _load_models(self):
        print("\n[Loading Models]")
        
        os.makedirs(self.models_path, exist_ok=True)
        
        model_files = {
            "rock": "norwegian_rock.json",
            "church": "fisherman_cabin.json",
            "turf_house": "fisherman_cabin.json",
            "waterfall": "norwegian_rock.json",
            "geyser": "norwegian_rock.json",
            "boat": "fjord_boat.json"
        }
        
        for obj_type, filename in model_files.items():
            filepath = os.path.join(self.models_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.model_cache[obj_type] = {
                        'vertices': np.array(data['vertices'], dtype=np.float32),
                        'indices': np.array(data['indices'], dtype=np.uint32)
                    }
                    print(f"  {obj_type}: OK")
                except Exception as e:
                    print(f"  {obj_type}: FAILED ({e})")
            else:
                print(f"  {obj_type}: NOT FOUND")
                
    def _load_scene(self):
        if os.path.exists(self.scene_path):
            try:
                with open(self.scene_path, 'r') as f:
                    self.objects = json.load(f)
                print(f"\n[Scene Loaded: {len(self.objects)} objects]")
            except:
                self.objects = []
                
    def _save_scene(self):
        with open(self.scene_path, 'w') as f:
            json.dump(self.objects, f, indent=2)
        print(f"[Scene Saved: {len(self.objects)} objects]")
        
    def _update_camera(self):
        rad_yaw = math.radians(self.camera_yaw)
        rad_pitch = math.radians(self.camera_pitch)
        
        self.camera_target[0] = self.camera_pos[0] + self.camera_distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        self.camera_target[1] = self.camera_pos[1] + self.camera_distance * math.sin(rad_pitch)
        self.camera_target[2] = self.camera_pos[2] + self.camera_distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        
    def _get_terrain_height(self, x, z):
        if self.elevation_data is None:
            return 0
            
        scale = 30.0
        gx = int(x / scale)
        gz = int(z / scale)
        
        h, w = self.elevation_data.shape
        if 0 <= gx < w and 0 <= gz < h:
            return self.elevation_data[gz, gx]
        return 0
        
    def _place_object(self, mouse_x, mouse_y):
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        win_x = float(mouse_x)
        win_y = float(viewport[3] - mouse_y)
        
        near = gluUnProject(win_x, win_y, 0.0, modelview, projection, viewport)
        far = gluUnProject(win_x, win_y, 1.0, modelview, projection, viewport)
        
        ray_origin = np.array(near)
        ray_dir = np.array(far) - ray_origin
        ray_len = np.linalg.norm(ray_dir)
        if ray_len > 0:
            ray_dir = ray_dir / ray_len
        
        if abs(ray_dir[1]) < 0.001:
            return
            
        t = -ray_origin[1] / ray_dir[1]
        if t < 0:
            return
            
        hit = ray_origin + ray_dir * t
        
        bounds = self.terrain_bounds
        if (bounds['min'][0] <= hit[0] <= bounds['max'][0] and
            bounds['min'][2] <= hit[2] <= bounds['max'][2]):
            
            terrain_h = self._get_terrain_height(hit[0], hit[2])
            
            obj = {
                "type": self.object_types[self.selected_type],
                "position": [float(hit[0]), float(terrain_h), float(hit[2])],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1]
            }
            self.objects.append(obj)
            print(f"Placed {obj['type']} at ({hit[0]:.0f}, {terrain_h:.0f}, {hit[2]:.0f})")
            
    def _draw_terrain(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.terrain_vbo)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 24, None)
        
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 24, ctypes.c_void_p(12))
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.terrain_ibo)
        
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
        glDrawElements(GL_TRIANGLES, self.terrain_vertex_count, GL_UNSIGNED_INT, None)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
    def _draw_objects(self):
        for obj in self.objects:
            pos = obj["position"]
            obj_type = obj["type"]
            
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            glScalef(100, 100, 100)
            
            if obj_type == "rock":
                glColor3f(0.4, 0.4, 0.38)
            elif obj_type == "church":
                glColor3f(0.95, 0.95, 0.9)
            elif obj_type == "turf_house":
                glColor3f(0.3, 0.35, 0.2)
            elif obj_type == "waterfall":
                glColor3f(0.7, 0.85, 0.95)
            elif obj_type == "geyser":
                glColor3f(0.6, 0.7, 0.5)
            elif obj_type == "boat":
                glColor3f(0.3, 0.25, 0.2)
            
            if obj_type in self.model_cache:
                mesh = self.model_cache[obj_type]
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, mesh['vertices'])
                glDrawElements(GL_TRIANGLES, len(mesh['indices']), GL_UNSIGNED_INT, mesh['indices'])
                glDisableClientState(GL_VERTEX_ARRAY)
            else:
                self._draw_placeholder(obj_type)
                
            glPopMatrix()
            
    def _draw_placeholder(self, obj_type):
        if obj_type == "rock":
            self._draw_box(1, 0.8, 1)
        elif obj_type == "church":
            self._draw_box(1.2, 2.0, 1.2)
        elif obj_type == "turf_house":
            self._draw_box(1.5, 0.8, 1.5)
        elif obj_type == "waterfall":
            self._draw_box(0.3, 2.0, 0.3)
        elif obj_type == "geyser":
            self._draw_cone(0.5, 0.8)
        elif obj_type == "boat":
            self._draw_box(1, 0.3, 0.5)
            
    def _draw_box(self, w, h, d):
        hw, hh, hd = w/2, h/2, d/2
        
        glBegin(GL_QUADS)
        glVertex3f(-hw, -hh, -hd)
        glVertex3f(hw, -hh, -hd)
        glVertex3f(hw, hh, -hd)
        glVertex3f(-hw, hh, -hd)
        
        glVertex3f(-hw, -hh, hd)
        glVertex3f(hw, -hh, hd)
        glVertex3f(hw, hh, hd)
        glVertex3f(-hw, hh, hd)
        
        glVertex3f(-hw, hh, -hd)
        glVertex3f(hw, hh, -hd)
        glVertex3f(hw, hh, hd)
        glVertex3f(-hw, hh, hd)
        
        glVertex3f(-hw, -hh, -hd)
        glVertex3f(hw, -hh, -hd)
        glVertex3f(hw, -hh, hd)
        glVertex3f(-hw, -hh, hd)
        
        glVertex3f(-hw, -hh, -hd)
        glVertex3f(-hw, hh, -hd)
        glVertex3f(-hw, hh, hd)
        glVertex3f(-hw, -hh, hd)
        
        glVertex3f(hw, -hh, -hd)
        glVertex3f(hw, hh, -hd)
        glVertex3f(hw, hh, hd)
        glVertex3f(hw, -hh, hd)
        glEnd()
        
    def _draw_cone(self, radius, height):
        segments = 16
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height/2, 0)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(radius * math.cos(angle), -height/2, radius * math.sin(angle))
        glEnd()
        
    def _print_controls(self):
        print("\n" + "=" * 60)
        print("ICELAND SCENE EDITOR")
        print("=" * 60)
        print("Camera:")
        print("  W/S/A/D  - Move camera")
        print("  SHIFT    - Move up")
        print("  CTRL     - Move down")
        print("  Mouse    - Look around (hold right button)")
        print("")
        print("Objects (Iceland Theme):")
        print("  1-Rock   2-Church   3-TurfHouse   4-Waterfall   5-Geyser   6-Boat")
        print("  Click    - Place object")
        print("  Ctrl+Z   - Undo")
        print("  Ctrl+S   - Save scene")
        print("")
        print("Display:")
        print("  F        - Toggle wireframe")
        print("  H        - Toggle help")
        print("  ESC      - Exit")
        print("=" * 60)
        
    def run(self):
        self.init()
        
        clock = pygame.time.Clock()
        running = True
        mouse_captured = False
        
        while running:
            dt = clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_1:
                        self.selected_type = 0
                        print(f"Selected: {self.object_types[0]}")
                    elif event.key == K_2:
                        self.selected_type = 1
                        print(f"Selected: {self.object_types[1]}")
                    elif event.key == K_3:
                        self.selected_type = 2
                        print(f"Selected: {self.object_types[2]}")
                    elif event.key == K_4:
                        self.selected_type = 3
                        print(f"Selected: {self.object_types[3]}")
                    elif event.key == K_5:
                        self.selected_type = 4
                        print(f"Selected: {self.object_types[4]}")
                    elif event.key == K_6:
                        self.selected_type = 5
                        print(f"Selected: {self.object_types[5]}")
                    elif event.key == K_f:
                        self.wireframe = not self.wireframe
                        print(f"Wireframe: {'ON' if self.wireframe else 'OFF'}")
                    elif event.key == K_h:
                        self.show_help = not self.show_help
                    elif event.key == K_s and pygame.key.get_mods() & KMOD_CTRL:
                        self._save_scene()
                    elif event.key == K_z and pygame.key.get_mods() & KMOD_CTRL:
                        if self.objects:
                            removed = self.objects.pop()
                            print(f"Removed: {removed['type']}")
                            
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1 and not mouse_captured:
                        self._place_object(*event.pos)
                    elif event.button == 3:
                        mouse_captured = True
                        pygame.event.set_grab(True)
                        pygame.mouse.set_visible(False)
                        
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 3:
                        mouse_captured = False
                        pygame.event.set_grab(False)
                        pygame.mouse.set_visible(True)
                        
                elif event.type == MOUSEMOTION:
                    if mouse_captured:
                        dx, dy = event.rel
                        self.camera_yaw += dx * self.rotate_speed
                        self.camera_pitch = max(-89, min(89, self.camera_pitch + dy * self.rotate_speed))
                        self._update_camera()
                        
            keys = pygame.key.get_pressed()
            speed = self.move_speed * (3 if keys[K_LSHIFT] else 1) * dt
            
            rad_yaw = math.radians(self.camera_yaw)
            forward = np.array([math.sin(rad_yaw), 0, math.cos(rad_yaw)])
            right = np.array([math.cos(rad_yaw), 0, -math.sin(rad_yaw)])
            
            if keys[K_w]:
                self.camera_pos += forward * speed
            if keys[K_s]:
                self.camera_pos -= forward * speed
            if keys[K_a]:
                self.camera_pos -= right * speed
            if keys[K_d]:
                self.camera_pos += right * speed
            if keys[K_LSHIFT]:
                self.camera_pos[1] += speed
            if keys[K_LCTRL]:
                self.camera_pos[1] -= speed
                
            self._update_camera()
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            glLoadIdentity()
            gluLookAt(
                self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                self.camera_target[0], self.camera_target[1], self.camera_target[2],
                0, 1, 0
            )
            
            self._draw_terrain()
            self._draw_objects()
            
            pygame.display.flip()
            
            fps = int(clock.get_fps())
            obj_type = self.object_types[self.selected_type]
            pygame.display.set_caption(f"Iceland Scene Editor | FPS: {fps} | Objects: {len(self.objects)} | Type: {obj_type}")
            
        pygame.quit()


if __name__ == "__main__":
    editor = NorwayFjordEditor()
    editor.run()
