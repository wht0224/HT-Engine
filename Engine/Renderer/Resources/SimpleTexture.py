class SimpleTexture:
    def __init__(self, texture_id, width=0, height=0):
        self.texture_id = texture_id
        self.width = width
        self.height = height
    
    def bind(self, unit=0):
        from OpenGL.GL import glActiveTexture, glBindTexture, GL_TEXTURE0, GL_TEXTURE_2D
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
    
    def unbind(self):
        from OpenGL.GL import glActiveTexture, glBindTexture, GL_TEXTURE0, GL_TEXTURE_2D
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def update(self):
        pass
    
    def destroy(self):
        from OpenGL.GL import glDeleteTextures
        glDeleteTextures([self.texture_id])
