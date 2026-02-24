import numpy as np

class OBJLoader:
    """
    Simple OBJ Loader for loading 3D models.
    Supports vertices, normals, and UVs.
    Ignores materials for now.
    """
    
    @staticmethod
    def load(file_path):
        """
        Load OBJ file and return a dictionary with mesh data.
        Returns:
            dict: {'vertices': np.array, 'normals': np.array, 'uvs': np.array, 'indices': np.array}
        """
        vertices = []
        normals = []
        uvs = []
        
        # Faces usually store indices (1-based)
        # We need to flatten them for OpenGL (triangle soup or indexed draw)
        # For simplicity, let's just create a triangle soup (unoptimized but robust)
        final_vertices = []
        final_normals = []
        final_uvs = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    values = line.split()
                    if not values: continue
                    
                    if values[0] == 'v':
                        vertices.append(list(map(float, values[1:4])))
                    elif values[0] == 'vn':
                        normals.append(list(map(float, values[1:4])))
                    elif values[0] == 'vt':
                        uvs.append(list(map(float, values[1:3])))
                    elif values[0] == 'f':
                        # Handle faces (triangles or quads)
                        # Format: v/vt/vn
                        face_indices = []
                        for v in values[1:]:
                            w = v.split('/')
                            # OBJ is 1-based
                            vi = int(w[0]) - 1
                            vti = int(w[1]) - 1 if len(w) > 1 and w[1] else -1
                            vni = int(w[2]) - 1 if len(w) > 2 and w[2] else -1
                            face_indices.append((vi, vti, vni))
                            
                        # Triangulate if quad
                        triangles = []
                        if len(face_indices) == 3:
                            triangles.append(face_indices)
                        elif len(face_indices) == 4:
                            triangles.append([face_indices[0], face_indices[1], face_indices[2]])
                            triangles.append([face_indices[0], face_indices[2], face_indices[3]])
                            
                        for tri in triangles:
                            for vi, vti, vni in tri:
                                final_vertices.append(vertices[vi])
                                if vti >= 0:
                                    final_uvs.append(uvs[vti])
                                else:
                                    final_uvs.append([0, 0])
                                if vni >= 0:
                                    final_normals.append(normals[vni])
                                else:
                                    final_normals.append([0, 1, 0])
                                    
            return {
                'vertices': np.array(final_vertices, dtype=np.float32),
                'normals': np.array(final_normals, dtype=np.float32),
                'uvs': np.array(final_uvs, dtype=np.float32)
            }
            
        except Exception as e:
            print(f"Error loading OBJ: {e}")
            return None
