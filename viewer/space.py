import sys
from pathlib import Path
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
import open3d as o3d

def load_geometry(path: str):
    ext = path.split(".")[-1].lower()

    if ext == "pts":
        points = np.loadtxt(path, dtype=np.float32)[:, :3]

        center = points.mean(axis=0)
        points = points - center

        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        return "points", points.astype(np.float32)

    elif ext in ["ply", "obj"]:
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        center = vertices.mean(axis=0)
        vertices = vertices - center

        scale = np.max(np.linalg.norm(vertices, axis=1))
        if scale > 0:
            vertices = vertices / scale

        return "mesh", vertices.astype(np.float32), faces.astype(np.int32)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def init_opengl(width: int, height: int) -> None:
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(3.0)

    glClearColor(0.08, 0.08, 0.10, 1.0)

def draw_axes(length: float = 1.2) -> None:
    glLineWidth(2.0)
    glBegin(GL_LINES)

    glColor3f(1.0, 0.2, 0.2)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(length, 0.0, 0.0)

    glColor3f(0.2, 1.0, 0.2)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, length, 0.0)

    glColor3f(0.2, 0.4, 1.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, length)

    glEnd()

def draw_wire_cube(size: float = 2.2) -> None:
    s = size / 2.0

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

def draw_points(points: np.ndarray):
    glColor3f(0.3, 0.8, 1.0)
    glBegin(GL_POINTS)
    for p in points:
        glVertex3f(*p)
    glEnd()

def draw_mesh(vertices, faces):
    glColor3f(0.7, 0.9, 1.0)
    glBegin(GL_TRIANGLES)
    for face in faces:
        for idx in face:
            glVertex3f(*vertices[idx])
    glEnd()

def main(path=None):
    DEFAULT_FILE = "./output/chair_mesh.ply"

    if path is None:
        path = DEFAULT_FILE

    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = load_geometry(path)

    if data[0] == "points":
        points = data[1]
        vertices, faces = None, None
    else:
        vertices, faces = data[1], data[2]
        points = None

    pygame.init()
    pygame.display.set_mode((1000, 800), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Viewer")

    init_opengl(1000, 800)

    yaw, pitch, distance = 0, 20, 3
    dragging = False
    last_mouse = (0, 0)

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    distance -= 0.2
                elif event.button == 5:
                    distance += 0.2

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.pos[0] - last_mouse[0], event.pos[1] - last_mouse[1]
                yaw += dx * 0.4
                pitch += dy * 0.4
                pitch = max(-89, min(89, pitch))
                last_mouse = event.pos

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(0, 0, -distance)
        glRotatef(pitch, 1, 0, 0)
        glRotatef(yaw, 0, 1, 0)

        draw_wire_cube()
        draw_axes()

        if points is not None:
            draw_points(points)
        else:
            draw_mesh(vertices, faces)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()