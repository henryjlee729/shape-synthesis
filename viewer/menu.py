import pygame
import pygame_gui
import viewer as viewer
from generate_points import generate_chairs
from mesh_generator import generate_mesh

GENERATED_FILE = '../test/chair_0.pts'

def draw_status_screen(screen, title_font, info_font, title_text, subtitle_text):
    screen.fill((20, 20, 30))  # clear screen with dark background

    # main title text
    title = title_font.render(title_text, True, (255, 255, 255))
    title_rect = title.get_rect(center=(500, 320))
    screen.blit(title, title_rect)

    # subtitle text
    subtitle = info_font.render(subtitle_text, True, (190, 190, 205))
    subtitle_rect = subtitle.get_rect(center=(500, 380))
    screen.blit(subtitle, subtitle_rect)

    # loading indicator
    loading = info_font.render('Please wait...', True, (150, 200, 255))
    loading_rect = loading.get_rect(center=(500, 430))
    screen.blit(loading, loading_rect)

    pygame.display.flip()  # update screen immediately

def run_menu():
    pygame.init()

    width, height = 1000, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Shape Synthesis')

    manager = pygame_gui.UIManager((width, height))  # handles UI elements
    clock = pygame.time.Clock()

    # fonts for UI text
    title_font = pygame.font.SysFont('Arial', 42, bold=True)
    info_font = pygame.font.SysFont('Arial', 22)

    # button placement (centered horizontally)
    generate_rect = pygame.Rect(width // 2 - 100, height // 2 + 100, 200, 60)

    # create "Generate" button
    generate_button = pygame_gui.elements.UIButton(
        relative_rect=generate_rect,
        text='Generate',
        manager=manager,
    )

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0  # frame time (for UI updates)

        for event in pygame.event.get():

            # close window
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

            # pass events to pygame_gui system
            manager.process_events(event)

            # handle button click
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == generate_button:

                    # show generation status
                    draw_status_screen(
                        screen,
                        title_font,
                        info_font,
                        'Generating...',
                        'Creating a new chair point cloud'
                    )
                    pygame.event.pump()  # keep window responsive

                    pts_file = generate_chairs(1)

                    # show mesh building status
                    draw_status_screen(
                        screen,
                        title_font,
                        info_font,
                        'Building...',
                        'Converting point cloud to mesh'
                    )
                    pygame.event.pump()

                    mesh_file = generate_mesh(pts_file)
                    return mesh_file  # exit menu and return result

        manager.update(time_delta)

        screen.fill((20, 20, 30))  # redraw background

        # main title
        title = title_font.render('Neural 3D Shape Generator', True, (255, 255, 255))
        title_rect = title.get_rect(center=(width // 2, height // 2 - 140))
        screen.blit(title, title_rect)

        # subtitle description
        subtitle = info_font.render('Generate a new chair and build its mesh', True, (190, 190, 205))
        subtitle_rect = subtitle.get_rect(center=(width // 2, height // 2 - 40))
        screen.blit(subtitle, subtitle_rect)

        # show output file path
        output_text = info_font.render(f'Point output: {GENERATED_FILE}', True, (150, 200, 255))
        output_rect = output_text.get_rect(center=(width // 2, height // 2 + 10))
        screen.blit(output_text, output_rect)

        manager.draw_ui(screen)  # draw button + UI elements
        pygame.display.flip()   # update screen

    pygame.quit()
    return None

if __name__ == '__main__':
    file_path = run_menu()

    # if generation succeeded, open viewer
    if file_path is not None:
        viewer.main(file_path)