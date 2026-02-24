# -*- coding: utf-8 -*-
"""
材质资源类，用于管理3D模型的材质属性
"""

from Engine.Math import Vector3

class Material:
    """材质资源类，用于管理3D模型的材质属性"""
    
    def __init__(self):
        """
        初始化材质
        """
        self.name = "DefaultMaterial"
        self.base_color = Vector3(0.8, 0.8, 0.8)
        self.roughness = 0.5
        self.metallic = 0.0
        self.ao = 1.0
        self.alpha = 1.0
        self.emissive = Vector3(0, 0, 0)
        self.emissive_strength = 0.0
        
        self.base_color_texture = None
        self.roughness_texture = None
        self.metallic_texture = None
        self.normal_texture = None
        self.ao_texture = None
        self.emissive_texture = None
        
        self.base_color_image = None
        self.roughness_image = None
        self.metallic_image = None
        self.normal_image = None
        self.ao_image = None
        self.emissive_image = None
        
        self.double_sided = False
        self.wireframe = False
        self.blend_mode = "opaque"
        self.alpha_test = False
        self.alpha_cutoff = 0.5
        
        self.cast_shadows = True
        self.receive_shadows = True
        
        self.shader = None
        self.shader_name = "pbr"
        
        self.is_dirty = True
    
    def set_color(self, color):
        self.base_color = color
        self.is_dirty = True
    
    def set_roughness(self, roughness):
        self.roughness = max(0.0, min(1.0, roughness))
        self.is_dirty = True
    
    def set_metallic(self, metallic):
        self.metallic = max(0.0, min(1.0, metallic))
        self.is_dirty = True
    
    def set_ao(self, ao):
        self.ao = max(0.0, min(1.0, ao))
        self.is_dirty = True
    
    def set_emissive(self, emissive, strength=1.0):
        self.emissive = emissive
        self.emissive_strength = max(0.0, strength)
        self.is_dirty = True
    
    def set_double_sided(self, double_sided):
        self.double_sided = double_sided
        self.is_dirty = True
    
    def set_wireframe(self, wireframe):
        self.wireframe = wireframe
        self.is_dirty = True
    
    BLEND_MODE_OPAQUE = "opaque"
    BLEND_MODE_TRANSPARENT = "transparent"
    BLEND_MODE_ADDITIVE = "additive"
    BLEND_MODE_MULTIPLY = "multiply"
    
    def set_transparency(self, transparency):
        self.alpha = max(0.0, min(1.0, transparency))
        self.is_dirty = True
    
    def set_blend_mode(self, blend_mode):
        valid_modes = ["opaque", "transparent", "additive", "multiply"]
        if blend_mode in valid_modes:
            self.blend_mode = blend_mode
            self.is_dirty = True
    
    def set_shader(self, shader):
        self.shader = shader
        self.is_dirty = True
    
    def set_shader_name(self, shader_name):
        self.shader_name = shader_name
        self.is_dirty = True
    
    def update(self):
        if not self.is_dirty:
            return

        if not hasattr(self, 'image_textures_uploaded'):
            self.image_textures_uploaded = False

        if (not self.image_textures_uploaded and 
            self.base_color_image is not None and 
            self.base_color_texture is None):
            try:
                self._upload_image_textures_to_gpu()
                self.image_textures_uploaded = True
            except Exception as e:
                print(f"材质更新: 无法上传纹理到GPU (可能在没有OpenGL上下文时): {e}")

        if self.base_color_texture:
            self.base_color_texture.update()
        if self.roughness_texture:
            self.roughness_texture.update()
        if self.metallic_texture:
            self.metallic_texture.update()
        if self.normal_texture:
            self.normal_texture.update()
        if self.ao_texture:
            self.ao_texture.update()
        if self.emissive_texture:
            self.emissive_texture.update()

        if self.shader:
            self._update_shader_params()

    def _upload_image_textures_to_gpu(self):
        from .SimpleTexture import SimpleTexture
        
        if self.base_color_image is not None and self.base_color_texture is None:
            try:
                if hasattr(self.base_color_image, 'shape') and len(self.base_color_image.shape) >= 2:
                    height, width = self.base_color_image.shape[0:2]
                    if len(self.base_color_image.shape) == 3:
                        channels = self.base_color_image.shape[2]
                        if channels == 3:
                            import numpy as np
                            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                            rgba_image[:, :, :3] = self.base_color_image
                            rgba_image[:, :, 3] = 255
                            image_data = rgba_image
                        elif channels == 4:
                            image_data = self.base_color_image
                        else:
                            import numpy as np
                            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                            rgba_image[:, :, 0] = self.base_color_image[:, :, 0] if channels > 0 else 0
                            rgba_image[:, :, 1] = self.base_color_image[:, :, 0] if channels > 0 else 0
                            rgba_image[:, :, 2] = self.base_color_image[:, :, 0] if channels > 0 else 0
                            rgba_image[:, :, 3] = 255
                            image_data = rgba_image
                    else:
                        import numpy as np
                        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                        rgba_image[:, :, 0] = self.base_color_image
                        rgba_image[:, :, 1] = self.base_color_image
                        rgba_image[:, :, 2] = self.base_color_image
                        rgba_image[:, :, 3] = 255
                        image_data = rgba_image

                    from OpenGL.GL import glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glTexParameterf, glGetFloatv, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, glGenerateMipmap, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_MAX_ANISOTROPY, GL_MAX_TEXTURE_MAX_ANISOTROPY
                    texture_id = glGenTextures(1)
                    if texture_id == 0:
                        print("警告：无法创建纹理ID，可能是因为OpenGL上下文未初始化")
                        return
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                    try:
                        max_anisotropy = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY)
                        if isinstance(max_anisotropy, (list, tuple)):
                            max_anisotropy = float(max_anisotropy[0])
                        elif hasattr(max_anisotropy, 'shape') and max_anisotropy.size > 0:
                            max_anisotropy = float(max_anisotropy.item())
                        else:
                            max_anisotropy = float(max_anisotropy)
                        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, max_anisotropy)
                    except:
                        pass
                    glGenerateMipmap(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, 0)

                    from .SimpleTexture import SimpleTexture
                    self.base_color_texture = SimpleTexture(texture_id, width, height)
                    print(f"成功上传基础颜色纹理到GPU: {width}x{height}")
                elif hasattr(self.base_color_image, 'size'):
                    pil_image = self.base_color_image
                    if pil_image.mode not in ('RGB', 'RGBA'):
                        pil_image = pil_image.convert('RGBA')

                    image_array = np.array(pil_image)
                    height, width = image_array.shape[0:2]

                    if len(image_array.shape) == 3:
                        channels = image_array.shape[2]
                        if channels == 3:
                            import numpy as np
                            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                            rgba_image[:, :, :3] = image_array
                            rgba_image[:, :, 3] = 255
                            image_data = rgba_image
                        elif channels == 4:
                            image_data = image_array
                        else:
                            import numpy as np
                            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                            rgba_image[:, :, 0] = image_array[:, :, 0] if channels > 0 else 0
                            rgba_image[:, :, 1] = image_array[:, :, 0] if channels > 0 else 0
                            rgba_image[:, :, 2] = image_array[:, :, 0] if channels > 0 else 0
                            rgba_image[:, :, 3] = 255
                            image_data = rgba_image
                    else:
                        import numpy as np
                        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                        rgba_image[:, :, 0] = image_array
                        rgba_image[:, :, 1] = image_array
                        rgba_image[:, :, 2] = image_array
                        rgba_image[:, :, 3] = 255
                        image_data = rgba_image

                    from OpenGL.GL import glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glTexParameterf, glGetFloatv, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, glGenerateMipmap, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_MAX_ANISOTROPY, GL_MAX_TEXTURE_MAX_ANISOTROPY
                    texture_id = glGenTextures(1)
                    if texture_id == 0:
                        print("警告：无法创建纹理ID，可能是因为OpenGL上下文未初始化")
                        return
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                    try:
                        max_anisotropy = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY)
                        if isinstance(max_anisotropy, (list, tuple)):
                            max_anisotropy = float(max_anisotropy[0])
                        elif hasattr(max_anisotropy, 'shape') and max_anisotropy.size > 0:
                            max_anisotropy = float(max_anisotropy.item())
                        else:
                            max_anisotropy = float(max_anisotropy)
                        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, max_anisotropy)
                    except:
                        pass
                    glGenerateMipmap(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, 0)

                    from .SimpleTexture import SimpleTexture
                    self.base_color_texture = SimpleTexture(texture_id, width, height)
                    print(f"成功上传基础颜色纹理到GPU: {width}x{height}")
                else:
                    print(f"基础颜色图像格式不正确，无法上传到GPU: {type(self.base_color_image)}")
            except Exception as e:
                print(f"上传基础颜色纹理到GPU失败: {e}")
                import traceback
                traceback.print_exc()

        if self.roughness_image is not None and self.roughness_texture is None:
            try:
                if hasattr(self.roughness_image, 'shape') and len(self.roughness_image.shape) >= 2:
                    height, width = self.roughness_image.shape[0:2]
                    if len(self.roughness_image.shape) == 2:
                        import numpy as np
                        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                        rgba_image[:, :, 0] = self.roughness_image
                        rgba_image[:, :, 1] = self.roughness_image
                        rgba_image[:, :, 2] = self.roughness_image
                        rgba_image[:, :, 3] = 255
                        image_data = rgba_image
                    else:
                        image_data = self.roughness_image

                    from OpenGL.GL import glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T
                    texture_id = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                    glBindTexture(GL_TEXTURE_2D, 0)

                    from .SimpleTexture import SimpleTexture
                    self.roughness_texture = SimpleTexture(texture_id, width, height)
            except Exception as e:
                print(f"上传粗糙度纹理到GPU失败: {e}")

        if self.metallic_image is not None and self.metallic_texture is None:
            try:
                if hasattr(self.metallic_image, 'shape') and len(self.metallic_image.shape) >= 2:
                    height, width = self.metallic_image.shape[0:2]
                    if len(self.metallic_image.shape) == 2:
                        import numpy as np
                        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                        rgba_image[:, :, 0] = self.metallic_image
                        rgba_image[:, :, 1] = self.metallic_image
                        rgba_image[:, :, 2] = self.metallic_image
                        rgba_image[:, :, 3] = 255
                        image_data = rgba_image
                    else:
                        image_data = self.metallic_image

                    from OpenGL.GL import glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T
                    texture_id = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                    glBindTexture(GL_TEXTURE_2D, 0)

                    from .SimpleTexture import SimpleTexture
                    self.metallic_texture = SimpleTexture(texture_id, width, height)
            except Exception as e:
                print(f"上传金属度纹理到GPU失败: {e}")
        
        self.is_dirty = False
    
    def _update_shader_params(self):
        if not self.shader:
            return
        
        if isinstance(self.shader, int):
            from OpenGL.GL import (
                glUseProgram, glGetUniformLocation,
                glUniform3f, glUniform1f, glUniform1i
            )
            
            program = self.shader
            glUseProgram(program)
            
            loc = glGetUniformLocation(program, "u_baseColor")
            if loc != -1:
                glUniform3f(loc, self.base_color.x, self.base_color.y, self.base_color.z)

            loc = glGetUniformLocation(program, "u_roughness")
            if loc != -1:
                glUniform1f(loc, self.roughness)

            loc = glGetUniformLocation(program, "u_metallic")
            if loc != -1:
                glUniform1f(loc, self.metallic)

            loc = glGetUniformLocation(program, "u_ao")
            if loc != -1:
                glUniform1f(loc, self.ao)

            loc = glGetUniformLocation(program, "u_alpha")
            if loc != -1:
                glUniform1f(loc, self.alpha)

            loc = glGetUniformLocation(program, "u_emissive")
            if loc != -1:
                glUniform3f(loc, self.emissive.x, self.emissive.y, self.emissive.z)

            loc = glGetUniformLocation(program, "u_emissiveStrength")
            if loc != -1:
                glUniform1f(loc, self.emissive_strength)
            
            texture_unit = 0
            if self.base_color_texture:
                loc = glGetUniformLocation(program, "u_baseColorTexture")
                if loc != -1:
                    glUniform1i(loc, texture_unit)
                texture_unit += 1
            if self.roughness_texture:
                loc = glGetUniformLocation(program, "u_roughnessTexture")
                if loc != -1:
                    glUniform1i(loc, texture_unit)
                texture_unit += 1
            if self.metallic_texture:
                loc = glGetUniformLocation(program, "u_metallicTexture")
                if loc != -1:
                    glUniform1i(loc, texture_unit)
                texture_unit += 1
            if self.normal_texture:
                loc = glGetUniformLocation(program, "u_normalTexture")
                if loc != -1:
                    glUniform1i(loc, texture_unit)
                texture_unit += 1
            if self.ao_texture:
                loc = glGetUniformLocation(program, "u_aoTexture")
                if loc != -1:
                    glUniform1i(loc, texture_unit)
                texture_unit += 1
            if self.emissive_texture:
                loc = glGetUniformLocation(program, "u_emissiveTexture")
                if loc != -1:
                    glUniform1i(loc, texture_unit)
                texture_unit += 1
        elif isinstance(self.shader, dict):
            pass
        elif hasattr(self.shader, 'set_vec3'):
            self.shader.set_vec3("u_baseColor", self.base_color)
            self.shader.set_float("u_roughness", self.roughness)
            self.shader.set_float("u_metallic", self.metallic)
            self.shader.set_float("u_ao", self.ao)
            self.shader.set_float("u_alpha", self.alpha)
            self.shader.set_vec3("u_emissive", self.emissive)
            self.shader.set_float("u_emissiveStrength", self.emissive_strength)

            texture_unit = 0
            if self.base_color_texture:
                self.shader.set_int("u_baseColorTexture", texture_unit)
                texture_unit += 1
            if self.roughness_texture:
                self.shader.set_int("u_roughnessTexture", texture_unit)
                texture_unit += 1
            if self.metallic_texture:
                self.shader.set_int("u_metallicTexture", texture_unit)
                texture_unit += 1
            if self.normal_texture:
                self.shader.set_int("u_normalTexture", texture_unit)
                texture_unit += 1
            if self.ao_texture:
                self.shader.set_int("u_aoTexture", texture_unit)
                texture_unit += 1
            if self.emissive_texture:
                self.shader.set_int("u_emissiveTexture", texture_unit)
                texture_unit += 1
    
    def bind(self):
        self.update()
        
        if self.shader:
            if isinstance(self.shader, int):
                from OpenGL.GL import glUseProgram
                glUseProgram(self.shader)
            elif hasattr(self.shader, 'bind'):
                self.shader.bind()
        
        texture_unit = 0
        if self.base_color_texture:
            self.base_color_texture.bind(texture_unit)
            texture_unit += 1
        if self.roughness_texture:
            self.roughness_texture.bind(texture_unit)
            texture_unit += 1
        if self.metallic_texture:
            self.metallic_texture.bind(texture_unit)
            texture_unit += 1
        if self.normal_texture:
            self.normal_texture.bind(texture_unit)
            texture_unit += 1
        if self.ao_texture:
            self.ao_texture.bind(texture_unit)
            texture_unit += 1
        if self.emissive_texture:
            self.emissive_texture.bind(texture_unit)
            texture_unit += 1
        
        self._set_render_state()
    
    def unbind(self):
        if self.base_color_texture:
            self.base_color_texture.unbind()
        if self.roughness_texture:
            self.roughness_texture.unbind()
        if self.metallic_texture:
            self.metallic_texture.unbind()
        if self.normal_texture:
            self.normal_texture.unbind()
        if self.ao_texture:
            self.ao_texture.unbind()
        if self.emissive_texture:
            self.emissive_texture.unbind()
        
        if self.shader:
            if isinstance(self.shader, int):
                from OpenGL.GL import glUseProgram
                glUseProgram(0)
            elif hasattr(self.shader, 'unbind'):
                self.shader.unbind()
    
    def _set_render_state(self):
        from OpenGL.GL import (
            glEnable, glDisable, glCullFace, glPolygonMode,
            GL_DEPTH_TEST, GL_CULL_FACE, GL_BACK,
            GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            GL_ONE, GL_ZERO, GL_FRONT_AND_BACK, GL_LINE, GL_FILL
        )
        
        if self.blend_mode == "opaque":
            glEnable(GL_DEPTH_TEST)
        else:
            glEnable(GL_DEPTH_TEST)
        
        if self.double_sided:
            glDisable(GL_CULL_FACE)
        else:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        
        if self.blend_mode == "transparent":
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        elif self.blend_mode == "additive":
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        elif self.blend_mode == "multiply":
            glEnable(GL_BLEND)
            glBlendFunc(GL_ZERO, GL_SRC_COLOR)
        else:
            glDisable(GL_BLEND)
        
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def destroy(self):
        if self.base_color_texture:
            self.base_color_texture.destroy()
        if self.roughness_texture:
            self.roughness_texture.destroy()
        if self.metallic_texture:
            self.metallic_texture.destroy()
        if self.normal_texture:
            self.normal_texture.destroy()
        if self.ao_texture:
            self.ao_texture.destroy()
        if self.emissive_texture:
            self.emissive_texture.destroy()
        pass
    
    @property
    def color(self):
        return self.base_color
    
    @color.setter
    def color(self, value):
        self.base_color = value
        self.is_dirty = True
