import numpy as np
import pygame
import open3d as o3d
from pathlib import Path
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
from mesh_generator import generate_mesh

def load_geometry(path: str):
    ext = path.split('.')[-1].lower()  # Get file extension

    # If input is .pts → convert to mesh first
    if ext == 'pts':
        path = generate_mesh(path)
        ext = 'ply'

    # Load mesh formats
    if ext in ['ply', 'obj']:
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()  # Needed for proper rendering

        # Convert Open3D structures → numpy arrays
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        # Center mesh at origin
        center = vertices.mean(axis=0)
        vertices = vertices - center

        # Normalize scale (fit inside unit sphere)
        scale = np.max(np.linalg.norm(vertices, axis=1))
        if scale > 0:
            vertices = vertices / scale

        return 'mesh', vertices.astype(np.float32), faces.astype(np.int32)

    # Unsupported file type
    raise ValueError(f'Unsupported file type: {ext}')

def init_opengl(width: int, height: int) -> None:
    glViewport(0, 0, width, height)

    # Set up projection (camera lens)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 0.1, 100.0)

    # Switch to model view (object transforms)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Enable depth testing (so closer objects block farther ones)
    glEnable(GL_DEPTH_TEST)

    # Enable smooth points (for point rendering, if used)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(3.0)

    # Background color (dark gray)
    glClearColor(0.02, 0.02, 0.04, 1.0)

def draw_axes(length: float = 1.2) -> None:
    glLineWidth(2.0)
    glBegin(GL_LINES)

    # X-axis (red)
    glColor3f(1.0, 0.2, 0.2)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(length, 0.0, 0.0)

    # Y-axis (green)
    glColor3f(0.2, 1.0, 0.2)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, length, 0.0)

    # Z-axis (blue)
    glColor3f(0.2, 0.4, 1.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, length)

    glEnd()

def draw_wire_cube(size: float = 2.2) -> None:
    s = size / 2.0

    # List of line segment endpoints
    edges = [
        (-s, -s, -s), ( s, -s, -s),
        ( s, -s, -s), ( s,  s, -s),
        ( s,  s, -s), (-s,  s, -s),
        (-s,  s, -s), (-s, -s, -s),

        (-s, -s,  s), ( s, -s,  s),
        ( s, -s,  s), ( s,  s,  s),
        ( s,  s,  s), (-s,  s,  s),
        (-s,  s,  s), (-s, -s,  s),

        (-s, -s, -s), (-s, -s,  s),
        ( s, -s, -s), ( s, -s,  s),
        ( s,  s, -s), ( s,  s,  s),
        (-s,  s, -s), (-s,  s,  s),
    ]

    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(1.0)

    glBegin(GL_LINES)
    for v in edges:
        glVertex3f(*v)
    glEnd()

def draw_mesh(vertices, faces):
    # Filled mesh
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    glColor3f(0.65, 0.65, 0.70)

    glBegin(GL_TRIANGLES)
    for face in faces:
        for idx in face:
            glVertex3f(*vertices[idx])
    glEnd()

    glDisable(GL_POLYGON_OFFSET_FILL)

    # Triangle edges on top
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glLineWidth(1.0)
    glColor3f(0.08, 0.08, 0.08)

    glBegin(GL_TRIANGLES)
    for face in faces:
        for idx in face:
            glVertex3f(*vertices[idx])
    glEnd()

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

def main(path=None):
    DEFAULT_FILE = './test/output.ply'

    if path is None:
        path = DEFAULT_FILE

    # Ensure file exists
    if not Path(path).exists():
        raise FileNotFoundError(f'File not found: {path}')

    # Load mesh
    _, vertices, faces = load_geometry(path)

    # Initialize pygame + OpenGL window
    pygame.init()
    pygame.display.set_mode((1000, 800), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Viewer')

    init_opengl(1000, 800)

    # Camera controls
    yaw, pitch, distance = 0, 20, 3
    dragging = False
    last_mouse = (0, 0)

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():

            # Close window
            if event.type == pygame.QUIT:
                running = False

            # ESC key exits
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            # Mouse controls
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click → rotate
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:  # Scroll up → zoom in
                    distance -= 0.2
                elif event.button == 5:  # Scroll down → zoom out
                    distance += 0.2

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False

            # Mouse drag → rotate camera
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.pos[0] - last_mouse[0], event.pos[1] - last_mouse[1]
                yaw += dx * 0.4
                pitch += dy * 0.4
                pitch = max(-89, min(89, pitch))  # prevent flipping
                last_mouse = event.pos

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Apply camera transforms
        glTranslatef(0, 0, -distance)
        glRotatef(pitch, 1, 0, 0)
        glRotatef(yaw, 0, 1, 0)

        # Draw scene
        draw_wire_cube()
        draw_axes()
        draw_mesh(vertices, faces)

        pygame.display.flip()
        clock.tick(60)  # limit to 60 FPS

    pygame.quit()

if __name__ == '__main__':
    main()